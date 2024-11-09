import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

import pytest
import numpy as np
from src.t_test import pre_check_visual, pre_check_stats, ttest_1sample, ttest_individual, ttest_paired

# 샘플 데이터
target_list1 = np.random.normal(50, 10, 100)  # 평균 50, 표준편차 10, 크기 100
target_list2 = np.random.normal(55, 10, 100)  # 평균 55, 표준편차 10, 크기 100

# 1. pre_check_visual 테스트 (시각화는 자동화된 테스트가 어렵기 때문에 여기서는 생략)
def test_pre_check_visual():
    try:
        pre_check_visual(target_list1)
    except Exception as e:
        pytest.fail(f"pre_check_visual 함수가 오류를 발생시켰습니다: {e}")

# 2. pre_check_stats 테스트
def test_pre_check_stats():
    try:
        pre_check_stats(target_list1)
        pre_check_stats(target_list1, target_list2)
    except Exception as e:
        pytest.fail(f"pre_check_stats 함수가 오류를 발생시켰습니다: {e}")

# 3. ttest_1sample 테스트
def test_ttest_1sample():
    compare_number = 50  # 비교 값
    try:
        ttest_1sample(target_list1, compare_number)
    except Exception as e:
        pytest.fail(f"ttest_1sample 함수가 오류를 발생시켰습니다: {e}")

# 4. ttest_individual 테스트
def test_ttest_individual():
    try:
        ttest_individual(target_list1, target_list2)
    except Exception as e:
        pytest.fail(f"ttest_individual 함수가 오류를 발생시켰습니다: {e}")

# 5. ttest_paired 테스트
def test_ttest_paired():
    try:
        ttest_paired(target_list1, target_list2)
    except Exception as e:
        pytest.fail(f"ttest_paired 함수가 오류를 발생시켰습니다: {e}")

# 추가적으로 예외 케이스 테스트
def test_empty_list():
    empty_list = []
    try:
        pre_check_stats(empty_list)
    except ValueError as e:
        assert str(e) == "Input data cannot be empty"
        
    try:
        ttest_1sample(empty_list, 50)
    except ValueError as e:
        assert str(e) == "Input data cannot be empty"
        
    try:
        ttest_individual(empty_list, empty_list)
    except ValueError as e:
        assert str(e) == "Input data cannot be empty"
