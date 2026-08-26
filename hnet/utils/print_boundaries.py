import torch


def print_boundary_tokens(input_ids, bpred_output, tokenizer):
    input_ids = input_ids.detach().cpu()

    # Cada elemento representa um grupo de tokens originais.
    # Inicialmente cada token é seu próprio grupo.
    stage_groups = [
        [[i] for i in range(input_ids.shape[1])]
        for _ in range(input_ids.shape[0])
    ]

    # Guardamos os IDs originais
    original_ids = [
        input_ids[b].tolist()
        for b in range(input_ids.shape[0])
    ]

    for stage, routing in enumerate(bpred_output):

        boundary_mask = routing.boundary_mask.detach().cpu()

        print("\n" + "=" * 70)
        print(f"STAGE {stage}")
        print("=" * 70)

        new_stage_groups = []

        for batch_idx in range(input_ids.shape[0]):

            boundaries = boundary_mask[batch_idx].tolist()
            groups = stage_groups[batch_idx]

            # ---------------------------------------------------------
            # Remove BOS apenas da visualização
            # ---------------------------------------------------------

            offset = 1 if (
                input_ids[batch_idx, 0].item() == tokenizer.bos_idx
            ) else 0

            display_boundaries = boundaries[offset:]
            display_groups = groups[offset:]

            print("\nChunks between boundaries:")

            current_group = []

            for group, boundary in zip(
                display_groups,
                display_boundaries,
            ):

                current_group.extend(group)

                if not boundary:
                    continue

                # -----------------------------------------------------
                # IDs dos tokens originais
                # -----------------------------------------------------

                chunk_ids = [
                    original_ids[batch_idx][idx]
                    for idx in current_group
                ]

                # -----------------------------------------------------
                # Tokens reais do tokenizer
                # -----------------------------------------------------

                chunk_tokens = tokenizer.convert_ids_to_tokens(
                    chunk_ids
                )

                # Texto reconstruído
                chunk_text = tokenizer.decode(
                    chunk_ids,
                    skip_special_tokens=False,
                )

                original_start = current_group[0]
                original_end = current_group[-1] + 1

                print(
                    f"[{original_start:3d} -> {original_end:3d}]: "
                    f"{chunk_text!r}"
                )

                print(
                    f"    tokens: {chunk_tokens}"
                )

                print(
                    f"    ids:    {chunk_ids}"
                )

                # Esse conjunto será um elemento no próximo estágio
                new_stage_groups.append(
                    current_group.copy()
                )

                current_group = []

            # ---------------------------------------------------------
            # Último chunk
            # ---------------------------------------------------------

            if current_group:

                chunk_ids = [
                    original_ids[batch_idx][idx]
                    for idx in current_group
                ]

                chunk_tokens = tokenizer.convert_ids_to_tokens(
                    chunk_ids
                )

                chunk_text = tokenizer.decode(
                    chunk_ids,
                    skip_special_tokens=False,
                )

                original_start = current_group[0]
                original_end = current_group[-1] + 1

                print(
                    f"[{original_start:3d} -> {original_end:3d}]: "
                    f"{chunk_text!r}"
                )

                print(
                    f"    tokens: {chunk_tokens}"
                )

                print(
                    f"    ids:    {chunk_ids}"
                )

                new_stage_groups.append(
                    current_group.copy()
                )

            print(
                f"\nTokens/chunks neste estágio: "
                f"{len(display_groups)}"
            )

            print(
                f"Boundaries: "
                f"{sum(display_boundaries)}"
            )

            # ---------------------------------------------------------
            # IMPORTANTE:
            # new_stage_groups acima precisa ser separado por batch.
            # ---------------------------------------------------------

        # -------------------------------------------------------------
        # Reconstruir os grupos para cada batch
        # -------------------------------------------------------------

        next_stage_groups = []

        for batch_idx in range(input_ids.shape[0]):

            boundaries = boundary_mask[batch_idx].tolist()
            groups = stage_groups[batch_idx]

            offset = 1 if (
                input_ids[batch_idx, 0].item() == tokenizer.bos_idx
            ) else 0

            display_boundaries = boundaries[offset:]
            display_groups = groups[offset:]

            groups_for_next_stage = []
            current = []

            for group, boundary in zip(
                display_groups,
                display_boundaries,
            ):

                current.extend(group)

                if boundary:
                    groups_for_next_stage.append(current)
                    current = []

            if current:
                groups_for_next_stage.append(current)

            next_stage_groups.append(groups_for_next_stage)

        stage_groups = next_stage_groups
