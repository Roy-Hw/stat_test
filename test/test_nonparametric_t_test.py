import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

import pytest
import numpy as np
from src.nonparametric_t_test import sign_test, wilcoxon_signed_rank_test, mann_whitney_u_test

# 샘플 데이터
group1 = np.random.randint(27000, 35000, size=24)
group2 = np.random.randint(28000, 36000, size=24)

# 1. sign_test 테스트 (부호검정)
def test_sign_test():
    try:
        sign_test(group1, group2)
    except Exception as e:
        pytest.fail(f"sign_test 함수가 오류를 발생시켰습니다: {e}")

# 2. wilcoxon_signed_rank_test 테스트 (윌콕슨 부호 순위 검정)
def test_wilcoxon_signed_rank_test():
    try:
        wilcoxon_signed_rank_test(group1, group2)
    except Exception as e:
        pytest.fail(f"wilcoxon_signed_rank_test 함수가 오류를 발생시켰습니다: {e}")

# 3. mann_whitney_u_test 테스트 (만-휘트니 U 검정)
def test_mann_whitney_u_test():
    try:
        mann_whitney_u_test(group1, group2)
    except Exception as e:
        pytest.fail(f"mann_whitney_u_test 함수가 오류를 발생시켰습니다: {e}")

# 추가적으로 예외 케이스 테스트 (빈 리스트에 대한 처리)
def test_empty_list():
    empty_list = []
    try:
        sign_test(empty_list, empty_list)
    except ValueError as e:
        assert str(e) == "Input data cannot be empty"
        
    try:
        wilcoxon_signed_rank_test(empty_list, empty_list)
    except ValueError as e:
        assert str(e) == "Input data cannot be empty"
        
    try:
        mann_whitney_u_test(empty_list, empty_list)
    except ValueError as e:
        assert str(e) == "Input data cannot be empty"
