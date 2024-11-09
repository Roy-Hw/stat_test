import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib
matplotlib.rc('font', family='AppleGothic')
matplotlib.rcParams['axes.unicode_minus'] = False 
import matplotlib.pyplot as plt
import seaborn as sns


def pre_check_visual(target_list):
    """t-test 수행을 위한 사전 가정을 만족하는지 시각화 방법으로 검정합니다."""
    fig, axes = plt.subplots(3, 1, figsize=(18, 12)) 

    # 히스토그램
    sns.histplot(target_list, kde=True, ax=axes[0])
    axes[0].set_title('히스토그램')  

    # 박스플롯
    sns.boxplot(x=target_list, ax=axes[1])  # x 인자에 target_list 전달
    axes[1].set_title('박스플롯')  

    # QQ plot
    stats.probplot(target_list, dist="norm", plot=axes[2])  
    axes[2].set_title('QQ plot')  # QQ plot 제목 설정

    # 레이아웃 조정 및 표시
    plt.tight_layout()  
    plt.show()


def pre_check_stats(target_list1, target_list2=None):
    """t-test 수행을 위한 사전 가정을 만족하는지 통계적 방법으로 검정합니다."""
    from scipy.stats import pearsonr
    # 1. 정규성 검정
    print("1. 정규성 검정")
    shapiro_stat1, shapiro_p_value1 = stats.shapiro(target_list1)
    ks_stat1, ks_p_value1 = stats.kstest(target_list1, 'norm') 
    print("Shapiro-Wilk Test for List1")
    print(f"Statistic: {shapiro_stat1:.4f}, p-value: {shapiro_p_value1:.4f}")
    print("Kolmogorov-Smirnov Test for List1")
    print(f"Statistic: {ks_stat1:.4f}, p-value: {ks_p_value1:.4f}")

    if target_list2 is None:
        print("\n대상 리스트가 하나임으로 독립성 및 등분산 검정을 수행하지 않습니다.")
    else:
        # List2에 대한 정규성 검정
        print("\n2. 정규성 검정 (List2)")
        shapiro_stat2, shapiro_p_value2 = stats.shapiro(target_list2)
        ks_stat2, ks_p_value2 = stats.kstest(target_list2, 'norm') 
        print("Shapiro-Wilk Test for List2")
        print(f"Statistic: {shapiro_stat2:.4f}, p-value: {shapiro_p_value2:.4f}")
        print("Kolmogorov-Smirnov Test for List2")
        print(f"Statistic: {ks_stat2:.4f}, p-value: {ks_p_value2:.4f}")
        
        # 2. 독립성 검정 (상관 분석)
        print("\n2. 독립성 검정")
        pearson_corr_stat, pearson_p_value = pearsonr(target_list1, target_list2)
        print("Pearson Correlation")
        print(f"Pearson correlation coefficient: {pearson_corr_stat:.4f}")
        print(f"Pearson p-value: {pearson_p_value:.4f}")      
        
        # 3. 등분산 검정
        print("\n3. 등분산 검정")
        bartlett_stat, bartlett_p_value = stats.bartlett(target_list1, target_list2)
        print("Bartlett's Test")
        print(f"Test Statistic: {bartlett_stat:.4f}")
        print(f"p-value: {bartlett_p_value:.4f}")

        levene_stat, levene_p_value = stats.levene(target_list1, target_list2)
        print("Levene's Test")
        print(f"Test Statistic: {levene_stat:.4f}")
        print(f"p-value: {levene_p_value:.4f}")


def ttest_1sample(target_list, compare_number):
    from scipy.stats import ttest_1samp
    tests = ['two-sided', 'greater', 'less']
    for test in tests:
        t_stat, p_value = ttest_1samp(target_list, compare_number, alternative=test)
        print(f"\n{test} 검정 결과:")
        print(f"t-statistic: {t_stat:.4f}, p-value: {p_value:.4f}")


def ttest_individual(target_list1, target_list2):
    from scipy.stats import ttest_ind
    tests = ['two-sided', 'greater', 'less']
    for test in tests:
        t_stat, p_value = ttest_ind(target_list1, target_list2, alternative=test)
        print(f"\n{test} 검정 결과:")
        print(f"t-statistic: {t_stat:.4f}, p-value: {p_value:.4f}")


def ttest_paired(target_list1, target_list2):
    from scipy.stats import ttest_rel
    tests = ['two-sided', 'greater', 'less']
    for test in tests:
        t_stat, p_value = ttest_rel(target_list1, target_list2, alternative=test)
        print(f"\n{test} 검정 결과:")
        print(f"t-statistic: {t_stat:.4f}, p-value: {p_value:.4f}")