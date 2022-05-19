from typing import Dict, Type

import sys
sys.path.append(".")

import os
os.environ['TZ'] = 'UTC'

MENU = ['Métricas Globais', 'Análise de Resíduos', 'Benchmark']

'''
PAGE_MAP: Dict[str, Type[Page]] = {
    MENU[0]: create_page1,
    MENU[1]: create_page2,
    MENU[2]: create_page3
}

initialize = create_initial_page

__all__ = ["PAGE_MAP"]
'''
