Label review, round 3 (2026-10-01)

Model yolo26s_stemcanker_v2/best.pt (epoch 22) was run on train + val with scripts/label_review.py.
Test was left out on purpose: its labels stay identical to v2, so v3 and v2 test scores compare directly.

runs/label_review_v3/      conf >= 0.3, any size: 849 candidates (train 696, val 153), 478 accepted
                           -> configs/label_fixes_v3.csv
runs/label_review_v3_big/  conf >= 0.1, box longer side >= 10% of the image, mined against the
                           labels above: 332 candidates, 87 accepted -> configs/label_fixes_v3_big.csv
                           (catches the large grey/tan necrotic cankers the model scores below 0.3)

sheet_XX.jpg   RED = candidate, GREEN = existing labels, yellow = id + model confidence.
accepted.txt   ids judged to be real lesions.

Rules (same as round 2):
  accepted  orange/brown ringed spots, grey/tan necrotic lesions with a clear edge (also large ones),
            white spots only where the same photo already labels identical white spots
  rejected  white scale-insect-like spots, areoles/spines, dried flower/bud remains, stem edges,
            black padding, glare, boxes spanning two lesions, round holes where the canker fell out,
            diffuse dead tissue without a clear edge, and every candidate round 2 already rejected

Build:   python scripts/apply_label_fixes.py --fixes configs/label_fixes_v2.csv configs/label_fixes_v3.csv configs/label_fixes_v3_big.csv --out ~/datasets/avx-dragonfruit/stemcanker_v3 --out-yaml configs/dataset_v3.yaml
Result:  train 7052 -> 7508 boxes, val 1109 -> 1218, test 889 (unchanged)
