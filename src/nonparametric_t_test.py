import scipy.stats as stats
import numpy as np

def sign_test(group1, group2):
    differences = group2 - group1
    positive = np.sum(differences > 0)
    negative = np.sum(differences < 0)
    n = positive + negative

    p_value_two_sided = 2 * min(stats.binom.cdf(positive, n, 0.5), 1 - stats.binom.cdf(positive - 1, n, 0.5))
    p_value_right = stats.binom.cdf(positive, n, 0.5)
    p_value_left = stats.binom.cdf(negative, n, 0.5)

    print("부호 검정 (양측 검정):")
    print(f"Positive Count: {positive}, Negative Count: {negative}, p-value: {p_value_two_sided:.3f}")
    print("\n부호 검정 (오른쪽 단측 검정):")
    print(f"Positive Count: {positive}, p-value: {p_value_right:.3f}")
    print("\n부호 검정 (왼쪽 단측 검정):")
    print(f"Negative Count: {negative}, p-value: {p_value_left:.3f}")

def wilcoxon_signed_rank_test(group1, group2):
    stat, p_value = stats.wilcoxon(group1, group2, alternative='two-sided')
    print("\n윌콕슨 부호 순위 검정 (양측 검정):")
    print(f"Test Statistic: {stat:.4f}, p-value: {p_value:.4f}")

def mann_whitney_u_test(group1, group2):
    stat, p_value = stats.mannwhitneyu(group1, group2, alternative='two-sided')
    print("\n만-휘트니 U 검정 (양측 검정):")
    print(f"U Statistic: {stat:.4f}, p-value: {p_value:.4f}")
