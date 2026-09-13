from typing import List, Union

def get_main_network_spec(arch_layout: List[Union[str, List]]) -> Union[str, List]:
    """
    Percorre o arch_layout da mesma forma que count_boundary_stages, mas
    retorna o spec (string) da rede central mais interna, ex.: 'T22', 'T26'
    ou 'Llama'.

    ["m4", ["T22"], "m4"]                            -> "T22"
    ["m4", ["T1m4", ["T26"], "m4T1"], "m4"]           -> "T26"
    ["m4", ["Llama"], "m4"]                           -> "Llama"
    ["m4", ["T1m4", ["Llama"], "m4T1"], "m4"]         -> "Llama"
    """
    layout = arch_layout
    while isinstance(layout, list) and len(layout) == 3:
        layout = layout[1]
    if isinstance(layout, list) and len(layout) >= 1:
        return layout[0]
    return layout

def count_boundary_stages(arch_layout: List[Union[str, List]]) -> int:
    n      = 0
    layout = arch_layout
    while isinstance(layout, list) and len(layout) == 3:
        n     += 1
        layout = layout[1]
    return n
