from __future__ import annotations
import logging
from pathlib import Path
import os
import sys
import time
import subprocess
import tempfile
from io import StringIO
import typing
import numpy as np
import torch
from huggingface_hub import snapshot_download
import requests
from Bio.Blast import NCBIXML  # pip install biopython
from rhofold.rhofold import RhoFold
from rhofold.config import rhofold_config
from rhofold.utils import get_device, save_ss2ct, timing
from rhofold.relax.relax import AmberRelaxation
from rhofold.utils.alphabet import get_features
from Bio import SeqIO

# NCBI BLAST URL
NCBI_BLAST_URL = "https://blast.ncbi.nlm.nih.gov/Blast.cgi"

# MUSCLE 실행 파일 경로 (질문에서 주신 경로 그대로 사용)
MUSCLE_BIN = "/home/connects/SCV_Models/Models/muscle/muscle"


def _read_single_seq_from_fasta(fasta_path: str) -> str:
    """
    단일 서열 FASTA라고 가정하고 첫 번째 시퀀스를 읽어서 반환합니다.
    """
    seq_lines: list[str] = []
    with open(fasta_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                # 이미 한 서열을 읽었다면 그 뒤 헤더부터는 무시
                if seq_lines:
                    break
                continue
            seq_lines.append(line)
    return "".join(seq_lines).upper()


def _run_ncbi_blastn(
    query_seq: str,
    hitlist_size: int = 100,
    db: str = "nt",
    program: str = "blastn",
    logger: typing.Optional[logging.Logger] = None,
) -> str:
    """
    NCBI BLAST Common URL API를 사용해서 blastn을 돌리고,
    결과(XML)를 문자열로 반환합니다.
    """
    # 1) 검색 제출 (CMD=Put)
    put_params = {
        "CMD": "Put",
        "PROGRAM": program,
        "DATABASE": db,
        "QUERY": query_seq,
        "HITLIST_SIZE": str(hitlist_size),
    }
    if logger:
        logger.info("Submitting BLASTN job to NCBI (DATABASE=nt)...")
    put_res = requests.post(NCBI_BLAST_URL, data=put_params)
    put_res.raise_for_status()

    rid = None
    rtoe = None
    for line in put_res.text.splitlines():
        line = line.strip()
        if line.startswith("RID ="):
            rid = line.split("=", 1)[1].strip()
        if line.startswith("RTOE ="):
            try:
                rtoe = int(line.split("=", 1)[1].strip())
            except Exception:
                rtoe = None

    if rid is None:
        raise RuntimeError("NCBI BLAST 응답에서 RID를 찾지 못했습니다.")

    if logger:
        logger.info(f"BLAST RID: {rid}, RTOE (sec): {rtoe}")

    # 2) 상태 폴링 (CMD=Get, FORMAT_OBJECT=SearchInfo)
    while True:
        time.sleep(10)
        status_params = {
            "CMD": "Get",
            "RID": rid,
            "FORMAT_OBJECT": "SearchInfo",
        }
        status_res = requests.get(NCBI_BLAST_URL, params=status_params)
        status_res.raise_for_status()

        status = None
        for line in status_res.text.splitlines():
            line = line.strip()
            if line.startswith("Status="):
                status = line.split("=", 1)[1].strip()
                break

        if logger:
            logger.info(f"NCBI BLASTN status: {status}")

        if status == "READY":
            break
        if status in ("FAILED", "UNKNOWN"):
            raise RuntimeError(f"BLAST 검색 실패 (Status={status}).")

    # 3) 결과 가져오기 (CMD=Get, FORMAT_TYPE=XML)
    get_params = {
        "CMD": "Get",
        "RID": rid,
        "FORMAT_TYPE": "XML",
    }
    res = requests.get(NCBI_BLAST_URL, params=get_params)
    res.raise_for_status()
    return res.text


def _extract_hit_seqs_from_blast_xml(
    xml_text: str,
    max_hits: int = 128,
    min_align_len: int = 30,
    logger: typing.Optional[logging.Logger] = None,
) -> list[str]:
    """
    BLAST XML을 파싱해서 hit 서열(부분 서열)을 추출합니다.
    alignment당 최선의 HSP 하나만 사용하고, align 길이가 짧은 것은 버립니다.
    """
    handle = StringIO(xml_text)
    blast_record = NCBIXML.read(handle)

    hit_seqs: list[str] = []
    for alignment in blast_record.alignments:
        best_hsp = None
        for hsp in alignment.hsps:
            if best_hsp is None or hsp.bits > best_hsp.bits:
                best_hsp = hsp
        if best_hsp is None:
            continue
        if best_hsp.align_length < min_align_len:
            continue

        # subject 서열에서 gap 제거
        hit_seq = best_hsp.sbjct.replace("-", "").upper()
        if not hit_seq:
            continue

        hit_seqs.append(hit_seq)
        if len(hit_seqs) >= max_hits:
            break

    if logger:
        logger.info(f"BLAST XML에서 {len(hit_seqs)}개의 hit 서열을 추출했습니다.")
    return hit_seqs


from Bio import SeqIO  # 파일 상단 import 추가 필요

from Bio import SeqIO  # 파일 상단 어딘가에 이미 있을 수도 있습니다.

def _run_muscle_and_write_a3m(
    query_seq,
    hit_seqs,
    output_a3m,
    logger=None,
):
    """
    1) query + hit 서열로 MUSCLE 정렬 수행
    2) 정렬 결과에서 'query에 "-"가 있는 column'만 제거 → 최종 길이를 원래 query 길이(예: 100)로 맞춤
    3) A3M에서는 항상 query를 첫 번째 서열로 기록
    """

    # BLAST hit 없음 → single sequence
    if not hit_seqs:
        if logger:
            logger.warning("BLAST hit이 없어 single-sequence A3M만 생성합니다.")
        with open(output_a3m, "w") as out_f:
            out_f.write(">query\n")
            out_f.write(query_seq + "\n")
        return

    import tempfile, subprocess
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        msa_input = tmpdir / "msa_input.fasta"
        msa_aligned = tmpdir / "msa_aligned.fasta"

        # ---------------------------
        # 1) query + hit FASTA 작성
        # ---------------------------
        with open(msa_input, "w") as f:
            f.write(">query\n")
            f.write(query_seq + "\n")
            for i, seq in enumerate(hit_seqs, start=1):
                f.write(f">hit_{i}\n")
                f.write(seq + "\n")

        # MUSCLE 입력 서열 로그
        if logger:
            logger.info("===== MUSCLE INPUT SEQUENCES =====")
            logger.info(f"Query (length={len(query_seq)}): {query_seq}")
            for idx, seq in enumerate(hit_seqs, start=1):
                logger.info(f"Hit {idx} (len={len(seq)}): {seq}")
            logger.info("===================================")

        # ------------------------------------------
        # 2) MUSCLE 실행
        # ------------------------------------------
        cmd = [
            "/home/connects/SCV_Models/Models/muscle/muscle",
            "--align",
            str(msa_input),
            "--output",
            str(msa_aligned),
        ]
        if logger:
            logger.info("Running MUSCLE: " + " ".join(cmd))
        subprocess.run(cmd, check=True)

        # ------------------------------------------
        # 3) MUSCLE 정렬 읽기
        #    → '진짜 query' 레코드 기준으로 gap column 제거
        # ------------------------------------------
        records = list(SeqIO.parse(str(msa_aligned), "fasta"))
        if not records:
            raise RuntimeError("MUSCLE 정렬 결과가 비어 있습니다.")

        # (1) MUSCLE output에서 query 레코드 찾기
        query_rec = None
        for rec in records:
            # FASTA 헤더가 '>query' 이므로 id 또는 description으로 찾는다
            if rec.id == "query" or rec.description.split()[0] == "query":
                query_rec = rec
                break

        if query_rec is None:
            raise RuntimeError("MUSCLE 정렬 결과에서 'query' 서열을 찾지 못했습니다.")

        aligned_query = str(query_rec.seq)
        L_align = len(aligned_query)

        # (2) query에서 gap이 아닌 column 인덱스만 남기기
        keep_indices = [i for i, ch in enumerate(aligned_query) if ch != "-"]
        L_nogap = len(keep_indices)

        if logger:
            logger.info(
                f"MUSCLE aligned length = {L_align}, "
                f"query_non_gap_length = {L_nogap}, "
                f"original_query_length = {len(query_seq)}"
            )

        # (3) 안전 검증: 탈갭 후 길이 == 원래 query 길이(100) 이어야 한다
        if L_nogap != len(query_seq):
            if logger:
                logger.error(
                    "탈갭 query 길이(%d)와 원래 query 길이(%d)가 다릅니다.",
                    L_nogap,
                    len(query_seq),
                )
                logger.error("aligned query: %s", aligned_query)
                logger.error("original query: %s", query_seq)
            raise RuntimeError("탈갭 query 길이와 원래 query 길이가 일치하지 않습니다.")

        def strip_cols(seq_str: str) -> str:
            """keep_indices에 해당하는 column만 남긴 서열 반환"""
            return "".join(seq_str[i] for i in keep_indices)

        # ------------------------------------------
        # 4) A3M 저장
        #    - 항상 query를 첫 번째 서열로
        #    - 모든 서열 길이가 정확히 len(query_seq) (=100) 인지 보증
        # ------------------------------------------
        with open(output_a3m, "w") as out_f:
            # 4-1. query 먼저 기록
            query_a3m = strip_cols(str(query_rec.seq))

            # query_a3m에서 gap 제거하면 원본 query와 일치해야 함
            if query_a3m.replace("-", "") != query_seq:
                if logger:
                    logger.error("A3M query와 원본 query가 일치하지 않습니다.")
                    logger.error("A3M query (degapped): %s", query_a3m.replace("-", ""))
                    logger.error("original query: %s", query_seq)
                raise RuntimeError("A3M query와 원본 query가 일치하지 않습니다.")

            out_f.write(">query\n")
            out_f.write(query_a3m + "\n")

            # 4-2. 나머지 hit 서열들
            for rec in records:
                if rec is query_rec:
                    continue  # 이미 썼음
                seq_str = str(rec.seq)
                if len(seq_str) != L_align:
                    raise RuntimeError("MSA 내부에서 서열 길이가 서로 다릅니다.")
                new_seq = strip_cols(seq_str)
                # 여기서 new_seq 길이는 반드시 len(query_seq) (예: 100)
                out_f.write(f">{rec.description}\n")
                out_f.write(new_seq + "\n")

        # ------------------------------------------
        # 5) 로그
        # ------------------------------------------
        if logger:
            logger.info(
                "A3M 파일 저장 완료 (query-gap column 제거 후 L=%d): %s",
                len(query_seq),
                output_a3m,
            )
            logger.info("===== MUSCLE ALIGNED SEQUENCES (RAW) =====")
            for rec in records:
                logger.info(f"{rec.id}: {rec.seq}")
            logger.info("=========================================")

def build_a3m_with_ncbi_blast(
    input_fas: str,
    output_a3m: str,
    logger: typing.Optional[logging.Logger] = None,
    hitlist_size: int = 100,
) -> None:
    """
    전체 파이프라인:
      1) input_fas에서 query 서열 읽기
      2) NCBI BLASTN(nt)로 유사 서열 검색
      3) BLAST XML에서 hit 서열 추출
      4) MUSCLE로 MSA 생성
      5) 결과를 A3M 파일로 저장
    """
    query_seq = _read_single_seq_from_fasta(input_fas)
    if logger:
        logger.info(f"Query length: {len(query_seq)}")

    xml_text = _run_ncbi_blastn(
        query_seq=query_seq,
        hitlist_size=hitlist_size,
        db="nt",
        program="blastn",
        logger=logger,
    )

    hit_seqs = _extract_hit_seqs_from_blast_xml(
        xml_text=xml_text,
        max_hits=hitlist_size,
        min_align_len=30,
        logger=logger,
    )

    _run_muscle_and_write_a3m(
        query_seq=query_seq,
        hit_seqs=hit_seqs,
        output_a3m=output_a3m,
        logger=logger,
    )


@torch.no_grad()
def main(config):
    """
    RhoFold Inference pipeline
    """

    os.makedirs(config.output_dir, exist_ok=True)

    logger = logging.getLogger('RhoFold Inference')
    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter('%(asctime)s - %(levelname)s: %(message)s')
    file_handler = logging.FileHandler(f'{config.output_dir}/log.txt', mode='w')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(logging.DEBUG)
    stream_handler.setFormatter(formatter)

    # 기존 핸들러 제거 후 새로 등록 (중복 방지)
    logger.handlers.clear()
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    logger.info('Constructing RhoFold')
    model = RhoFold(rhofold_config)

    logger.info(f'    downloading {config.ckpt}')
    try:
        snapshot_download(repo_id='cuhkaih/rhofold', local_dir=Path(config.ckpt).parent)
    except Exception as e:
        logger.info(
            f'    Error: Could not download the checkpoint from Hugging Face (cuhkaih/rhofold) '
            f'to {Path(config.ckpt).parent}.'
        )
        raise e

    logger.info(f'    loading {config.ckpt}')
    model.load_state_dict(torch.load(config.ckpt, map_location=torch.device('cpu'))['model'])
    model.eval()

    # Input seq, MSA
    logger.info(f"Input_fas {config.input_fas}")

    # if config.single_seq_pred:
    #     config.input_a3m = config.input_fas
    #     logger.info(
    #         "Input_a3m is None, the modeling will run using single sequence only (input_fas)"
    #     )

    if config.input_a3m is None:
        # 여기서 우리의 새 로직: NCBI BLAST + MUSCLE로 A3M 생성
        config.input_a3m = f'{config.output_dir}/seq.a3m'
        logger.info(
            "Input_a3m is None, building MSA via NCBI BLASTN (nt) + MUSCLE"
        )

        build_a3m_with_ncbi_blast(
            input_fas=config.input_fas,
            output_a3m=config.input_a3m,
            logger=logger,
            hitlist_size=100,
        )

        logger.info(f"Input_a3m {config.input_a3m}")

    else:
        logger.info(f"Input_a3m {config.input_a3m}")

    with timing('RhoFold Inference', logger=logger):
        config.device = get_device(config.device)
        logger.info(f'    Inference using device {config.device}')
        model = model.to(config.device)

        data_dict = get_features(config.input_fas, config.input_a3m)

        # Forward pass
        outputs = model(
            tokens=data_dict['tokens'].to(config.device),
            rna_fm_tokens=data_dict['rna_fm_tokens'].to(config.device),
            seq=data_dict['seq'],
        )

        output = outputs[-1]

        os.makedirs(config.output_dir, exist_ok=True)

        # Secondary structure, .ct format
        ss_prob_map = torch.sigmoid(output['ss'][0, 0]).data.cpu().numpy()
        ss_file = f'{config.output_dir}/ss.ct'
        save_ss2ct(ss_prob_map, data_dict['seq'], ss_file, threshold=0.5)

        # Dist prob map & Secondary structure prob map, .npz format
        npz_file = f'{config.output_dir}/results.npz'
        np.savez_compressed(
            npz_file,
            dist_n=torch.softmax(output['n'].squeeze(0), dim=0).data.cpu().numpy(),
            dist_p=torch.softmax(output['p'].squeeze(0), dim=0).data.cpu().numpy(),
            dist_c=torch.softmax(output['c4_'].squeeze(0), dim=0).data.cpu().numpy(),
            ss_prob_map=ss_prob_map,
            plddt=output['plddt'][0].data.cpu().numpy(),
        )

        # Save the prediction
        unrelaxed_model = f'{config.output_dir}/unrelaxed_model.pdb'

        # The last coords prediction
        node_cords_pred = output['cord_tns_pred'][-1].squeeze(0)
        model.structure_module.converter.export_pdb_file(
            data_dict['seq'],
            node_cords_pred.data.cpu().numpy(),
            path=unrelaxed_model,
            chain_id=None,
            confidence=output['plddt'][0].data.cpu().numpy(),
            logger=logger,
        )

    # Amber relaxation (옵션)
    # if config.relax_steps is not None:
    #     relax_steps = int(config.relax_steps)
    #     if relax_steps > 0:
    #         with timing(f'Amber Relaxation : {relax_steps} iterations', logger=logger):
    #             amber_relax = AmberRelaxation(max_iterations=relax_steps, logger=logger)
    #             relaxed_model = f'{config.output_dir}/relaxed_{relax_steps}_model.pdb'
    #             amber_relax.process(unrelaxed_model, relaxed_model)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--device",
        help="Default cpu. If GPUs are available, you can set --device cuda:<GPU_index> for faster prediction.",
        default=None,
    )
    parser.add_argument(
        "--ckpt",
        help="Path to the pretrained model, default ./pretrained/rhofold_pretrained_params.pt",
        default='./pretrained/rhofold_pretrained_params.pt',
    )
    parser.add_argument(
        "--input_fas",
        help="Path to the input fasta file. Valid nucleic acids in RNA sequence: A, U, G, C",
        required=True,
    )
    parser.add_argument(
        "--input_a3m",
        help=(
            "Path to the input msa file. Default None. "
            "If --input_a3m is not given (set to None), MSA will be generated automatically."
        ),
        default=None,
    )
    parser.add_argument(
        "--output_dir",
        help=(
            "Path to the output dir. "
            "3D prediction is saved in .pdb format. "
            "Distogram prediction is saved in .npz format. "
            "Secondary structure prediction is save in .ct format."
        ),
        required=True,
    )
    parser.add_argument(
        "--relax_steps",
        help="Num of steps for structure refinement, default 1000.",
        default=1000,
    )
    parser.add_argument(
        "--single_seq_pred",
        help=(
            "Default False. If --single_seq_pred is set to True, "
            "the modeling will run using single sequence only (input_fas)"
        ),
        default=False,
    )
    parser.add_argument(
        "--database_dpath",
        help="(더 이상 사용하지 않지만, 기존 인터페이스 호환을 위해 남김)",
        default='./database',
    )
    parser.add_argument(
        "--binary_dpath",
        help="(더 이상 사용하지 않지만, 기존 인터페이스 호환을 위해 남김)",
        default='./rhofold/data/bin',
    )

    args = parser.parse_args()
    main(args)
