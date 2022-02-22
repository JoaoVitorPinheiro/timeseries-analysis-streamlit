from pages.page0 import create_initial_page
from pages.page1 import create_page1
from pages.page2 import create_page2
from pages.page3 import create_page3
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
