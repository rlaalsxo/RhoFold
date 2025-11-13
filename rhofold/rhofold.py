# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import torch.nn as nn

from rhofold.model.embedders import MSAEmbedder, RecyclingEmbedder
from rhofold.model.e2eformer import E2EformerStack
from rhofold.model.structure_module import StructureModule
from rhofold.model.heads import DistHead, SSHead, pLDDTHead
from rhofold.utils.tensor_utils import add
from rhofold.utils import exists


class RhoFold(nn.Module):
    """The rhofold network"""

    def __init__(self, config):
        """Constructor function."""

        super().__init__()

        self.config = config

        self.msa_embedder = MSAEmbedder(
            **config.model.msa_embedder,
        )
        self.e2eformer = E2EformerStack(
            **config.model.e2eformer_stack,
        )
        self.structure_module = StructureModule(
            **config.model.structure_module,
        )
        self.recycle_embnet = RecyclingEmbedder(
            **config.model.recycling_embedder,
        )
        self.dist_head = DistHead(
            **config.model.heads.dist,
        )
        self.ss_head = SSHead(
            **config.model.heads.ss,
        )
        self.plddt_head = pLDDTHead(
            **config.model.heads.plddt,
        )


    def forward_cords(self, tokens, single_fea, pair_fea, seq):
        # 3단계: StructureModule offload 사용 여부 읽기
        use_offload = getattr(self.config.globals, "structure_offload", False)

        e2e_outputs = {
            "single": single_fea,
            "pair": pair_fea,
        }

        output = self.structure_module.forward(
            seq,
            tokens,
            e2e_outputs,
            _offload_inference=use_offload,
        )
        output['plddt'] = self.plddt_head(output['single'][-1])

        return output

    def forward_heads(self, pair_fea):

        output = {}
        output['ss'] = self.ss_head(pair_fea.float())
        output['p'], output['c4_'], output['n'] = self.dist_head(pair_fea.float())

        return output

    def forward_one_cycle(self, tokens, rna_fm_tokens, recycling_inputs, seq):
        '''
        Args:
            tokens: [bs, seq_len, c_z]
            rna_fm_tokens: [bs, seq_len, c_z]
        '''

        device = tokens.device

        # 1단계: chunk 최소 크기
        min_chunk_size = getattr(self.config.globals, "e2e_min_chunk_size", None)

        # 2단계: offload 사용 여부
        use_offload = getattr(self.config.globals, "e2e_offload", False)

        # chunk_size 정리 (None 또는 int)
        chunk_size = int(min_chunk_size) if min_chunk_size is not None else None

        msa_tokens_pert = tokens[:, :self.config.globals.msa_depth]

        msa_fea, pair_fea = self.msa_embedder.forward(
            tokens=msa_tokens_pert,
            rna_fm_tokens=rna_fm_tokens,
            is_BKL=True,
        )

        if exists(self.recycle_embnet) and exists(recycling_inputs):
            msa_fea_up, pair_fea_up = self.recycle_embnet(
                recycling_inputs['single_fea'],
                recycling_inputs['pair_fea'],
                recycling_inputs["cords_c1'"],
            )
            msa_fea[..., 0, :, :] += msa_fea_up
            pair_fea = add(pair_fea, pair_fea_up, inplace=False)

        msa_mask = torch.ones(msa_fea.shape[:3], device=device)
        pair_mask = torch.ones(pair_fea.shape[:3], device=device)

        # 2단계 핵심: offload 경로 vs 기존 경로 분기
        if use_offload and chunk_size is not None:
            # E2EformerStack._forward_offload 경로 사용 :contentReference[oaicite:0]{index=0}
            msa_fea, pair_fea, single_fea = self.e2eformer._forward_offload(
                input_tensors=[msa_fea, pair_fea],
                msa_mask=msa_mask,
                pair_mask=pair_mask,
                chunk_size=chunk_size,
            )
        else:
            # 1단계에서 쓰던 기존 forward 경로 (chunking만)
            msa_fea, pair_fea, single_fea = self.e2eformer(
                m=msa_fea,
                z=pair_fea,
                msa_mask=msa_mask,
                pair_mask=pair_mask,
                chunk_size=chunk_size,
            )

        output = self.forward_cords(tokens, single_fea, pair_fea, seq)
        output.update(self.forward_heads(pair_fea))

        recycling_outputs = {
            'single_fea': msa_fea[..., 0, :, :].detach(),
            'pair_fea': pair_fea.detach(),
            "cords_c1'": output["cords_c1'"][-1].detach(),
        }

        return output, recycling_outputs


    def forward(self,
                tokens,
                rna_fm_tokens,
                seq,
                **kwargs):

        """Perform the forward pass.

        Args:

        Returns:
        """

        recycling_inputs = None

        outputs = []
        for _r in range(self.config.model.recycling_embedder.recycles):
            output, recycling_inputs = \
                self.forward_one_cycle(tokens, rna_fm_tokens, recycling_inputs, seq)
            outputs.append(output)

        return outputs
