"""
boundary_probe.py — Analisa e salva boundary predictions do H-Net por posição.

Para cada texto de entrada, gera um arquivo .txt com:
  - Linha 1: byte | char | prob_stage0 | ind_stage0
  - Linha 2: (se 2 estágios) prob_stage1 | ind_stage1

Salvo em: tests/<model_name>/<frase_idx>/passo_<step>.txt
"""

import re
import unicodedata
from pathlib import Path

import torch

def _safe_dirname(text: str, max_len: int = 40) -> str:
    """
    Transforma um texto arbitrário num nome de diretório seguro:
    - Remove acentos
    - Mantém só alfanuméricos e underscores
    - Trunca em max_len caracteres
    """
    # Remove acentos (NFD → só ASCII)
    normalized = unicodedata.normalize("NFD", text)
    ascii_only = "".join(c for c in normalized if unicodedata.category(c) != "Mn")
    # Mantém alfanuméricos e espaços, troca o resto por _
    safe = re.sub(r"[^\w\s]", "_", ascii_only)
    safe = re.sub(r"\s+", "_", safe.strip())
    return safe[:max_len] if safe else "text"


def _char_repr(byte_val: int) -> str:
    """
    Representação legível de um byte:
    - Bytes printáveis ASCII → o caractere direto
    - Newline → <NL>
    - Tab → <TAB>
    - Carriage return → <CR>
    - Outros → <0xXX>
    """
    if byte_val == 0x0A:
        return "<NL>"
    elif byte_val == 0x09:
        return "<TAB>"
    elif byte_val == 0x0D:
        return "<CR>"
    elif 0x20 <= byte_val <= 0x7E:
        return chr(byte_val)
    else:
        return f"<0x{byte_val:02X}>"


@torch.no_grad()
def boundary_probe(
    model: torch.nn.Module,
    step: int,
    texts: list[str],
    out_root: str = "text",
    device: str = "cuda",
):

    was_training = model.training
    model.eval()

    try:
        for text_idx, text in enumerate(texts):

            # ---------------------------------------------------------
            # 1. Texto -> bytes UTF-8
            # ---------------------------------------------------------
            raw_bytes = text.encode("utf-8")

            byte_ids = torch.tensor(list(raw_bytes), dtype=torch.long, device=device).unsqueeze(0)

            # ---------------------------------------------------------
            # 2. Forward
            # ---------------------------------------------------------
            try:
                output = model(byte_ids)
            except Exception as e:
                print(
                    f"[boundary_probe] Erro no forward "
                    f"texto={text_idx}: {e}"
                )
                continue

            # ---------------------------------------------------------
            # 3. Verifica saída
            # ---------------------------------------------------------
            if not hasattr(output, "bpred_output") or not output.bpred_output:
                print(
                    "[boundary_probe] Modelo não retornou "
                    "bpred_output."
                )
                return

            # ---------------------------------------------------------
            # 4. Extrai cada stage
            # ---------------------------------------------------------
            stages = []

            for bpred in output.bpred_output:

                prob = bpred.boundary_prob[..., 1].float().cpu().reshape(-1)
                ind = bpred.boundary_mask.float().cpu().reshape(-1)

                stages.append((prob, ind))

            # ---------------------------------------------------------
            # 5. Diretório
            # ---------------------------------------------------------
            dir_name = _safe_dirname(text)

            out_dir = (Path(out_root) / dir_name)
            out_dir.mkdir(parents=True, exist_ok=True)

            out_path = (out_dir / f"passo_{step}.txt")

            # ---------------------------------------------------------
            # 6. Salva
            # ---------------------------------------------------------
            with open(out_path, "w", encoding="utf-8") as f:

                f.write(f"# passo: {step}\n")
                f.write(f"# texto: {repr(text)}\n")
                f.write(f"# bytes: {len(raw_bytes)}\n")
                f.write(f"# estagios: {len(stages)}\n")
                f.write("#\n")

                # Stage 0
                prob0, ind0 = stages[0]

                f.write(
                    "# [stage 0]\n"
                    "# pos | byte | char | boundary_prob | boundary_ind\n"
                )

                for pos, byte_val in enumerate(raw_bytes):

                    if pos >= len(prob0):
                        break

                    char_repr = _char_repr(byte_val)

                    f.write(
                        f"{pos} | "
                        f"0x{byte_val:02X} | "
                        f"{char_repr} | "
                        f"{prob0[pos].item():.5f} | "
                        f"{int(ind0[pos].item())}\n"
                    )

                # Stages seguintes
                for s_idx in range(1, len(stages)):

                    prob_s, ind_s = stages[s_idx]

                    f.write("\n")
                    f.write(f"# [stage {s_idx}]\n")
                    f.write(
                        "# chunk_pos | boundary_prob | boundary_ind\n"
                    )

                    for pos in range(len(prob_s)):
                        f.write(
                            f"{pos} | "
                            f"{prob_s[pos].item():.5f} | "
                            f"{int(ind_s[pos].item())}\n"
                        )

            print(
                f"[boundary_probe] "
                f"step={step} "
                f"text={text_idx} "
                f"-> {out_path}"
            )

    finally:
        if was_training:
            model.train()