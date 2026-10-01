Test-label audit (2026-10-01): is v3 really worse than v2, or do the frozen test labels penalise it?

test_audit.py mine   v2 + v3 detections on the test split that touch no label (conf >= 0.3, or >= 0.1 if the
                     box's longer side is >= 8% of the image), merged across models -> 199 candidates.
                     Sheets are BLIND: id only, no model, no confidence. Same accept/reject rules as round 3
                     (runs/label_review_v3/README.txt). 108 accepted (accepted.txt).
                     Accepted by source: only v2 23/59, only v3 35/54, both 50/86.
test_audit.py score  frozen test labels + 108 boxes -> ~/datasets/avx-dragonfruit/stemcanker_testaudit
                     (889 -> 997 boxes; the frozen stemcanker_v3 test split is untouched), then both models on both.

                         v2 frozen   v3 frozen   v2 audited   v3 audited
  mAP50                    0.385       0.368       0.429        0.426
  mAP50-95                 0.201       0.194       0.232        0.235
  best F1                  0.453       0.438       0.502        0.500
  precision @ 0.5          0.734       0.670       0.787        0.751
  false alarms/img @ 0.5   0.471       0.647       0.376        0.488

Result: on the audited labels v2 and v3 are tied. The frozen labels cost v3 more than v2 because more of v3's
unlabelled detections are real lesions.

The rest of v3's extra false alarms are mostly NOT unlabelled lesions. They are tight v3 boxes inside a much bigger
test label (at conf >= 0.5: v3 53 vs v2 34, median area 10% of the label). Many test labels include the yellow halo
around a spot, or put one box around a group of lesions. v3 boxes the lesion core only, so IoU < 0.5 and it counts
as a false alarm plus a missed lesion. That is a box-convention conflict in the labels, not a model error.
One test photo (the one with the round holes, candidates 163/166) also has several big boxes over bare stem.
