"""Quick PRM analysis: matched-coverage precision vs X-SGRV."""
import json
from sklearn.metrics import roc_auc_score
from scipy.stats import binomtest

qwen = json.load(open('/Users/aayanalwani/MATHAI/MATHAI/results/exp40_qwen_prm.json'))
se = json.load(open('/Users/aayanalwani/MATHAI/MATHAI/results/exp35_semantic_entropy.json'))
plur = {(r['bench'], r['id']): r.get('plurality_correct', False) for r in se['rows']}

XSGRV_K = {"math175": 96, "aime": 2, "cleanmath": 4}

for bench in ['math175', 'aime', 'cleanmath']:
    rows = qwen.get(bench, {})
    if not rows:
        continue
    n_labeled = sum(1 for k in plur if k[0] == bench)
    print(f'\n[{bench}] n_probs={len(rows)}, plur labels available: {n_labeled}')
    for agg in ['min', 'product', 'last', 'mean']:
        scores, labels = [], []
        for pid, pdata in rows.items():
            per = pdata.get('per_sample', [])
            if not per:
                continue
            s = max(float(x.get(agg, 0.5)) for x in per)
            lbl = plur.get((bench, pid))
            if lbl is None:
                continue
            scores.append(s)
            labels.append(lbl)
        if not scores:
            print(f'  {agg}: no scores')
            continue
        labels_wrong = [not x for x in labels]
        if len(set(labels_wrong)) == 2:
            auroc = float(roc_auc_score(labels_wrong, [-x for x in scores]))
        else:
            auroc = None
        n = len(scores)
        k = XSGRV_K[bench]
        sorted_idx = sorted(range(n), key=lambda i: -scores[i])
        top_k = sorted_idx[:k]
        matched_correct = sum(1 for i in top_k if labels[i])
        ci = binomtest(matched_correct, k).proportion_ci(0.95, 'exact')
        ba = f'{auroc:.3f}' if auroc is not None else 'n/a'
        print(f'  {agg:8s}: k={k}, prec={matched_correct}/{k}={matched_correct/k:.3f} '
              f'[{ci.low:.2f},{ci.high:.2f}] AUROC={ba}')
