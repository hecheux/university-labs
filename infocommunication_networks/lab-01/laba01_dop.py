import math
import numpy as np
import matplotlib.pyplot as plt


def trim_leading_zeros(bits):
    #Удаление ведущих нулей
    for i, bit in enumerate(bits):
        if bit == 1:
            return bits[i:]
    return [0]


def poly_mod(dividend, divisor):
    #Деление полиномов

    dividend = trim_leading_zeros(list(dividend))
    divisor = trim_leading_zeros(list(divisor))

    if divisor == [0]:
        raise ValueError("Делитель не может быть нулевым полиномом.")

    remainder = dividend[:]
    #1
    while remainder != [0] and len(remainder) >= len(divisor):
        shift = len(remainder) - len(divisor)
        aligned_divisor = divisor + [0] * shift
        remainder = [a ^ b for a, b in zip(remainder, aligned_divisor)]
        remainder = trim_leading_zeros(remainder)

    return remainder


def encode_crc(message_bits, generator_bits):
    #Формирование кодового слова

    r = len(generator_bits) - 1
    shifted_message = message_bits + [0] * r
    remainder = poly_mod(shifted_message, generator_bits)

    if len(remainder) < r:
        remainder = [0] * (r - len(remainder)) + remainder

    return message_bits + remainder


def generate_random_message(l):
    #Генерация случайного сообщения
    return np.random.randint(0, 2, l).tolist()


def generate_error_vector(n, p):
    # Генерирует вектор ошибок длины n. Вероятность ошибки в каждом бите равна p

    return (np.random.random(n) < p).astype(int).tolist()


def estimate_decoding_error_probability(generator_bits, l, p, epsilon):
    # Вероятность ошибки декодирования

    r = len(generator_bits) - 1
    n = l + r

    N = math.ceil(9 / (4 * (epsilon ** 2)))

    decoding_errors = 0

    for _ in range(N):

        message = generate_random_message(l)
        codeword = encode_crc(message, generator_bits)

        error_vector = generate_error_vector(n, p)
        received = [a ^ e for a, e in zip(codeword, error_vector)]
        syndrome = poly_mod(received, generator_bits)

        if syndrome == [0] and any(error_vector):
            decoding_errors += 1

    pe_hat = decoding_errors / N
    return pe_hat, N


def polynomial_to_string(bits):
    #Преобразует список коэффициентов в строку
    degree = len(bits) - 1
    terms = []

    for i, bit in enumerate(bits):
        if bit == 0:
            continue

        power = degree - i

        if power > 1:
            terms.append(f"x^{power}")
        elif power == 1:
            terms.append("x")
        else:
            terms.append("1")

    return " + ".join(terms) if terms else "0"


def plot_results(results, generator_bits):
    #Строит график зависимости оценки вероятности ошибки декодирования от p.
    plt.figure(figsize=(10, 6))

    l_values = sorted(set(item["l"] for item in results))

    for l in l_values:
        current = [item for item in results if item["l"] == l]
        current.sort(key=lambda x: x["p"])

        p_values = [item["p"] for item in current]
        pe_values = [item["Pe_hat"] for item in current]

        plt.plot(p_values, pe_values, marker="o", linewidth=2, label=f"l = {l}")

    plt.grid(True, linestyle="--", alpha=0.7)
    plt.xlabel("Вероятность ошибки в канале p")
    plt.ylabel("Оценка вероятности ошибки декодирования Pe")
    plt.title(f"Зависимость Pe от p, g(x) = {polynomial_to_string(generator_bits)}")
    plt.legend()
    plt.tight_layout()
    plt.show()


def run_research():
    generator_bits = [1, 1, 0, 1]
    l_values = [3, 4, 5]

    # Значения вероятности ошибки в канале
    p_values = [i / 10 for i in range(0, 11)]
    epsilon = 0.01

    results = []

    print(f"Порождающий многочлен: g(x) = {polynomial_to_string(generator_bits)}")
    print(f"Точность epsilon = {epsilon}")
    print()

    print(f"{'l':<8}{'p':<8}{'Pe':<18}{'N':<12}")
    print("-" * 46)

    for l in l_values:
        for p in p_values:
            pe_hat, N = estimate_decoding_error_probability(generator_bits=generator_bits, l=l, p=p, epsilon=epsilon)

            print(f"{l:<8}{p:<8.1f}{pe_hat:<18.6f}{N:<12}")

            results.append({
                "l": l,
                "p": p,
                "Pe_hat": pe_hat,
                "N": N
            })

        print()

    plot_results(results, generator_bits)


#DOPPPP


def dec_std(b, g):
    s = poly_mod(b, g)
    return s != [0]


def dec_alt(b, g, l):
    r = len(g) - 1

    mb = b[:l]
    bc = b[l:l + r]

    a_new = encode_crc(mb, g)
    bc_new = a_new[l:l + r]

    return bc != bc_new


def est_both(g, l, p, eps):
    #Оценка Pe для двух декодеров
    r = len(g) - 1
    n = l + r
    N = math.ceil(9 / (4 * eps ** 2))

    err_std = 0
    err_alt = 0
    diff = 0

    for _ in range(N):
        m = generate_random_message(l)
        a = encode_crc(m, g)

        e = generate_error_vector(n, p)
        b = [x ^ y for x, y in zip(a, e)]

        has_err = any(e)

        std_found = dec_std(b, g)
        alt_found = dec_alt(b, g, l)

        if has_err and not std_found:
            err_std += 1

        if has_err and not alt_found:
            err_alt += 1

        if std_found != alt_found:
            diff += 1

    return {
        "Pe_std": err_std / N,
        "Pe_alt": err_alt / N,
        "N": N,
        "diff": diff
    }


def plot_cmp(data, g):
    plt.figure(figsize=(12, 7))

    l_vals = sorted(set(x["l"] for x in data))

    for l in l_vals:
        cur = [x for x in data if x["l"] == l]
        cur.sort(key=lambda x: x["p"])

        p_vals = [x["p"] for x in cur]
        pe_std = [x["Pe_std"] for x in cur]
        pe_alt = [x["Pe_alt"] for x in cur]

        plt.plot(
            p_vals,
            pe_std,
            marker="o",
            linewidth=2,
            label=f"Типовой, l={l}"
        )

        plt.plot(
            p_vals,
            pe_alt,
            marker="s",
            linestyle="--",
            linewidth=2,
            label=f"Альт., l={l}"
        )

    plt.grid(True, linestyle="--", alpha=0.7)
    plt.xlabel("p")
    plt.ylabel("P̂e")
    plt.title(f"Сравнение декодеров, g(x) = {polynomial_to_string(g)}")
    plt.legend()
    plt.tight_layout()
    plt.show()


def run_alt():
    g = [1, 1, 0, 1]
    l_vals = [3, 4, 5]
    p_vals = [i / 10 for i in range(0, 11, 2)]
    eps = 0.01

    data = []

    print("Сравнение декодеров")
    print(f"g(x) = {polynomial_to_string(g)}")
    print(f"eps = {eps}")
    print()

    print(f"{'l':<6}{'p':<8}{'P̂e тип.':<16}{'P̂e альт.':<16}{'Различия':<14}{'N':<12}")
    print("-" * 70)

    for l in l_vals:
        for p in p_vals:
            res = est_both(g, l, p, eps)

            print(
                f"{l:<6}{p:<8.1f}{res['Pe_std']:<16.6f}"
                f"{res['Pe_alt']:<16.6f}{res['diff']:<14}{res['N']:<12}"
            )

            data.append({
                "l": l,
                "p": p,
                "Pe_std": res["Pe_std"],
                "Pe_alt": res["Pe_alt"],
                "diff": res["diff"],
                "N": res["N"]
            })
        print()

    plot_cmp(data, g)

    total_diff = sum(x["diff"] for x in data)
    print("-" * 70)
    print(f"Всего различий решений: {total_diff}")



if __name__ == "__main__":
    np.random.seed(42)
    run_alt()