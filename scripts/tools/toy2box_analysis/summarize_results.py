"""Print a compact summary of a registered analysis output (artifacts/experiments/registered_results.json).

Usage: python3 summarize_results.py <registered_results.json>
"""

import json
import sys

BINS = ("[0,45)", "[45,90)", "[90,135)", "[135,181)")
STAGES = ("approach", "grasp", "transport", "placement", "success")


def main():
    d = json.load(open(sys.argv[1]))
    cond = d["conditions"]
    print("condition  s101  s111  pooled  mean±sd(ddof1)      v3.1 stages a/g/t/p/s   success by |yaw| bin " + "/".join(BINS))
    for n, c in cond.items():
        m = c["models"]
        cs = c["condition_success"]
        st = c["v3_1_pooled_stages"]
        bins = c["close_by_abs_yaw_pooled"]
        yaw = " ".join(f"{bins[b]['successes']}/{bins[b]['trials']}" for b in BINS)
        print(f"{n:9}  {m['101']['successes']:4}  {m['111']['successes']:4}  {c['pooled_successes']:3}/200  "
              f"{100*cs['mean_rate']:5.1f}% ± {100*cs['sample_sd_rate_ddof1']:4.2f}%p   "
              f"{'/'.join(str(st[s]) for s in STAGES):22}  {yaw}")
    print()
    print("condition  [45,90) closed mode0/mode180/between  herr median   recovery: first failed -> recovered (retried, later secured); first secured -> success; never near")
    for n, c in cond.items():
        t = c["close_by_abs_yaw_pooled"]["[45,90)"]
        r = c["recovery_pooled"]
        med = t["heading_error_abs_median_deg"]
        print(f"{n:9}  {t['closed']:3}  {t['mode0']}/{t['mode180']}/{t['between']:<14}  {med if med is None else round(med, 1)!s:>8}°   "
              f"{r['first_failed']} -> {r['recoveries']} ({r['first_failed_retried_near']}, {r['first_failed_later_secured']}); "
              f"{r['first_secured']} -> {r['success_given_first_secured']}; {r['never_near']}")
    print()
    for n, comp in d["comparisons"].items():
        w, a = comp["pooled_whole"], comp["pooled_abs_yaw_le_90"]
        print(f"{n}: whole {w['first']['successes']}/{w['first']['trials']} vs {w['second']['successes']}/{w['second']['trials']} "
              f"p={w['p_one_sided']:.4f}; |yaw|<=90 {a['first']['successes']}/{a['first']['trials']} vs "
              f"{a['second']['successes']}/{a['second']['trials']} p={a['p_one_sided']:.4f}")
    print()
    for h, dec in d["preregistered_decisions"].items():
        print(f"{h}: {dec['verdict']}  rules={json.dumps(dec['rules'])}")
        if h == "H1_double_prime":
            p2 = dec["prediction_2_observed"]
            print(f"   prediction 2: between {p2['between']}/{p2['closed']} = {p2['between_fraction']:.3f} "
                  f"(limit {p2['half_baseline_fraction_limit']:.3f}), herr median {p2['heading_error_median_deg']}")
            print(f"   E2 specificity: E1 {dec['E2_specificity']['E1_pooled']} vs E2 {dec['E2_specificity']['E2_pooled']} ({dec['E2_specificity']['verdict']})")
        if h == "H3":
            rt = dec["recovery_test_vs_E0_5_of_102"]
            print(f"   recovery test: {rt['first']['successes']}/{rt['first']['trials']} vs 5/102 p={rt['p_one_sided']:.4f}")


if __name__ == "__main__":
    main()
