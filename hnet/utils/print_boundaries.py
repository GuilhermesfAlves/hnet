# def print_boundary_tokens(input_ids, bpred_output, tokenizer):
#     input_ids = input_ids.detach().cpu()
# 
#     for stage, routing in enumerate(bpred_output):
#         boundary_mask = routing.boundary_mask.detach().cpu()
# 
#         print("\n" + "=" * 60)
#         print(f"STAGE {stage}")
#         print("=" * 60)
# 
#         for batch_idx in range(input_ids.shape[0]):
#             ids = input_ids[batch_idx].tolist()
#             boundaries = boundary_mask[batch_idx].tolist()
# 
#             # Remove BOS
#             if ids and ids[0] == tokenizer.bos_idx:
#                 ids = ids[1:]
#                 boundaries = boundaries[1:]
#                 offset = 1
#             else:
#                 offset = 0
# 
#             print("\nChunks between boundaries:")
#             
#             last_idx = 0
#             for i, boundary in enumerate(boundaries):
#                 if not boundary:
#                     continue
#                 
#                 # Pega os tokens entre o último limite e o limite atual
#                 chunk_ids = ids[last_idx:i]
#                 
#                 try:
#                     chunk_text = tokenizer.decode(chunk_ids)
#                 except UnicodeDecodeError:
#                     chunk_text = repr(bytes(chunk_ids))
#                     
#                 print(f"[{last_idx + offset:3d} -> {i + offset:3d}]: {chunk_text!r}")
#                 
#                 # Atualiza o índice inicial para o próximo chunk
#                 last_idx = i
#             
#             # Imprime o último pedaço da frase (após o último boundary)
#             if last_idx < len(ids):
#                 chunk_ids = ids[last_idx:]
#                 try:
#                     chunk_text = tokenizer.decode(chunk_ids)
#                 except UnicodeDecodeError:
#                     chunk_text = repr(bytes(chunk_ids))
#                 print(f"[{last_idx + offset:3d} -> {len(ids) + offset:3d}]: {chunk_text!r}")
# 
#             print(
#                 f"\nTokens: {len(ids) + offset}"
#                 f" | Boundaries: {sum(boundaries)}"
#             )
import torch

def print_boundary_tokens(input_ids, bpred_output, tokenizer):
    input_ids = input_ids.detach().cpu()

    # Índices dos tokens na sequência original.
    # Cada estágio vai reduzir esse vetor.
    original_indices = torch.arange(
        input_ids.shape[1],
        dtype=torch.long,
    )

    for stage, routing in enumerate(bpred_output):

        boundary_mask = routing.boundary_mask.detach().cpu()

        print("\n" + "=" * 60)
        print(f"STAGE {stage}")
        print("=" * 60)

        for batch_idx in range(input_ids.shape[0]):

            ids = input_ids[batch_idx].tolist()
            boundaries = boundary_mask[batch_idx].tolist()

            # Índices originais correspondentes aos tokens deste estágio
            stage_indices = original_indices.tolist()

            # Remove BOS somente para exibição
            if ids and ids[0] == tokenizer.bos_idx:
                display_ids = ids[1:]
                display_boundaries = boundaries[1:]
                display_indices = stage_indices[1:]
            else:
                display_ids = ids
                display_boundaries = boundaries
                display_indices = stage_indices

            print("\nChunks between boundaries:")

            last_idx = 0

            for i, boundary in enumerate(display_boundaries):

                if not boundary:
                    continue

                chunk_ids = display_ids[last_idx:i]

                try:
                    chunk_text = tokenizer.decode(chunk_ids)
                except UnicodeDecodeError:
                    chunk_text = repr(bytes(chunk_ids))

                original_start = display_indices[last_idx]
                original_end = display_indices[i]

                print(
                    f"[{original_start:3d} -> {original_end:3d}]: "
                    f"{chunk_text!r}"
                )

                last_idx = i

            # Último chunk
            if last_idx < len(display_ids):

                chunk_ids = display_ids[last_idx:]

                try:
                    chunk_text = tokenizer.decode(chunk_ids)
                except UnicodeDecodeError:
                    chunk_text = repr(bytes(chunk_ids))

                original_start = display_indices[last_idx]

                if display_indices:
                    original_end = display_indices[-1] + 1
                else:
                    original_end = original_start

                print(
                    f"[{original_start:3d} -> {original_end:3d}]: "
                    f"{chunk_text!r}"
                )

            print(
                f"\nTokens: {len(display_ids)}"
                f" | Boundaries: {sum(display_boundaries)}"
            )

        # ---------------------------------------------------------
        # Os tokens selecionados neste estágio são a entrada
        # para o próximo estágio.
        # ---------------------------------------------------------

        selected = boundary_mask[0]

        original_indices = original_indices[selected]
