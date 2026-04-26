import math
import random
import numpy as np
import matplotlib.pyplot as plt

random.seed(15)
np.random.seed(15)

def generate_exp_intervals(lambda_rate, n):
    intervals = []
    for _ in range(n):
        u = random.random()
        while u == 0:
            u = random.random()
        x = -math.log(u) / lambda_rate
        intervals.append(x)
    return intervals

def marsaglia_normal(mean, sigma):
    while True:
        u1 = 2.0 * random.random() - 1.0
        u2 = 2.0 * random.random() - 1.0
        s = u1*u1 + u2*u2
        if 0 < s < 1:
            break
    factor = math.sqrt(-2.0 * math.log(s) / s)
    z0 = u1 * factor
    return mean + sigma * z0

def generate_normal_times(mean_ms, sigma_ms, n):
    times = []
    mean_sec = mean_ms / 1000.0
    sigma_sec = sigma_ms / 1000.0
    for _ in range(n):
        t = marsaglia_normal(mean_sec, sigma_sec)
        if t < 0:
            t = 0.0
        times.append(t)
    return times

N = 100000
lambda_rate = 13.0
mean_ms = 41.0
sigma_ms = 5.0

intervals = generate_exp_intervals(lambda_rate, N)
service_times = generate_normal_times(mean_ms, sigma_ms, N)

exp_mean_sample = np.mean(intervals)
exp_var_sample = np.var(intervals, ddof=1)
exp_theor_mean = 1.0 / lambda_rate
exp_theor_var = 1.0 / (lambda_rate**2)

norm_mean_sample = np.mean(service_times) * 1000
norm_var_sample = np.var(service_times, ddof=1) * 1e6
norm_theor_mean = mean_ms
norm_theor_var = sigma_ms**2

print("Характеристики для экспоненциальных интервалов (секунды)")
print(f"Мат ожидание: {exp_mean_sample:.6f} (теор. {exp_theor_mean:.6f})")
print(f"Выборочная дисперсия: {exp_var_sample:.6f} (теор. {exp_theor_var:.6f})")

print("\nХарактеристики для нормальных времён обработки (мс)")
print(f"Мат ожидание: {norm_mean_sample:.2f} мс (теор. {norm_theor_mean:.2f} мс)")
print(f"Выборочная дисперсия: {norm_var_sample:.2f} мс² (теор. {norm_theor_var:.2f} мс²)")

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.hist(intervals, bins=100, density=True, alpha=0.6, color='skyblue', label='Гистограмма')
x_exp = np.linspace(0, max(intervals), 200)
pdf_exp = lambda_rate * np.exp(-lambda_rate * x_exp)
plt.plot(x_exp, pdf_exp, 'r-', label='Теоретическая плотность')
plt.title('Плотность вероятности интервалов между приходами')
plt.xlabel('Время (с)')
plt.ylabel('Плотность')
plt.legend()

plt.subplot(1, 2, 2)
ecdf_exp = np.sort(intervals)
ecdf_y = np.arange(1, N+1) / N
plt.step(ecdf_exp, ecdf_y, where='post', label='Эмпирическая ФР')
plt.plot(x_exp, 1 - np.exp(-lambda_rate * x_exp), 'r-', label='Теоретическая ФР')
plt.title('Функция распределения интервалов')
plt.xlabel('Время (с)')
plt.ylabel('F(x)')
plt.legend()
plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 5))
service_sec = np.array(service_times)

plt.subplot(1, 2, 1)
plt.hist(service_sec, bins=100, density=True, alpha=0.6, color='lightgreen', label='Гистограмма')
x_norm = np.linspace(min(service_sec), max(service_sec), 200)
mean_sec = mean_ms/1000.0
sigma_sec = sigma_ms/1000.0
pdf_norm = (1/(sigma_sec * np.sqrt(2*np.pi))) * np.exp(-0.5*((x_norm - mean_sec)/sigma_sec)**2)
plt.plot(x_norm, pdf_norm, 'r-', label='Теоретическая плотность')
plt.title('Плотность вероятности времени обработки')
plt.xlabel('Время (с)')
plt.ylabel('Плотность')
plt.legend()

plt.subplot(1, 2, 2)
ecdf_norm = np.sort(service_sec)
ecdf_y = np.arange(1, N+1) / N
plt.step(ecdf_norm, ecdf_y, where='post', label='Эмпирическая ФР')
dx = x_norm[1] - x_norm[0]
theor_cdf_norm = np.cumsum(pdf_norm) * dx
theor_cdf_norm = theor_cdf_norm / theor_cdf_norm[-1]
plt.plot(x_norm, theor_cdf_norm, 'r-', label='Теоретическая ФР (интегрирование PDF)')
plt.title('Функция распределения времени обработки')
plt.xlabel('Время (с)')
plt.ylabel('F(x)')
plt.legend()
plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
sample_exp = np.sort(intervals)
n = len(sample_exp)
p = np.arange(1, n+1) / (n + 1)
theor_quantiles_exp = -np.log(1 - p) / lambda_rate
plt.scatter(theor_quantiles_exp, sample_exp, s=1, alpha=0.6, color='blue')
max_val = max(np.max(theor_quantiles_exp), np.max(sample_exp))
plt.plot([0, max_val], [0, max_val], 'r--', label='X')
plt.xlabel('Теоретические квантили')
plt.ylabel('Эмпирические квантили')
plt.title('QQ-plot для экспоненциального распределения')
plt.legend()
plt.subplot(1, 2, 2)
sample_norm = np.sort(service_sec)
def normal_quantile(p, mean, sigma):
    theor_sample = generate_normal_times(mean_ms, sigma_ms, n)
    theor_sample.sort()
    return theor_sample
theor_quantiles_norm = normal_quantile(p, mean_sec, sigma_sec)
plt.scatter(theor_quantiles_norm, sample_norm, s=1, alpha=0.6, color='green')
max_val_norm = max(np.max(theor_quantiles_norm), np.max(sample_norm))
plt.plot([0, max_val_norm], [0, max_val_norm], 'r--', label='Y')
plt.xlabel('Теоретические квантили')
plt.ylabel('Эмпирические квантили')
plt.title('QQ-plot для нормального распределения')
plt.legend()
plt.tight_layout()
plt.show()

def simulate_gateway(arrival_times, service_times, duration):

    n = len(arrival_times)
    departures = []
    last_departure = 0.0
    delays = []
    for i in range(n):
        if arrival_times[i] > duration:
            break
        start = max(arrival_times[i], last_departure)
        finish = start + service_times[i]
        departures.append(finish)
        last_departure = finish
        delays.append(finish - arrival_times[i])

    return delays

arrival_times = []
t = 0.0
for interval in intervals:
    t += interval
    arrival_times.append(t)

delays = simulate_gateway(arrival_times, service_times, duration=10.0)
if delays:
    prob_delay_gt_80ms = sum(1 for d in delays if d > 0.04) / len(delays)
    print(f"\nМоделирование")
    print(f"Всего пакетов за 10 с: {len(delays)}")
    print(f"Вероятность задержки > 40 мс: {prob_delay_gt_80ms:.4f} ({prob_delay_gt_80ms*100:.2f}%)")
else:
    print("Нет пакетов за 10 секунд.")

new_sigma_ms = 10.0
new_service_times = generate_normal_times(mean_ms, new_sigma_ms, N)

delays_new = simulate_gateway(arrival_times, new_service_times, duration=10.0)
if delays_new:
    prob_delay_gt_80ms_new = sum(1 for d in delays_new if d > 0.08) / len(delays_new)
    print(f"\nМоделирование (sigma = {new_sigma_ms} мс)")
    print(f"Всего пакетов за 10 с: {len(delays_new)}")
    print(f"Вероятность задержки > 80 мс: {prob_delay_gt_80ms_new:.4f} ({prob_delay_gt_80ms_new*100:.2f}%)")
else:
    print("Нет пакетов за 10 секунд.")

new_service_ms = np.array(new_service_times) * 1000
print(f"\nМат ожидание (новое sigma): {np.mean(new_service_ms):.2f} мс")
print(f"Выборочная дисперсия (новое sigma): {np.var(new_service_ms, ddof=1):.2f} мс²")