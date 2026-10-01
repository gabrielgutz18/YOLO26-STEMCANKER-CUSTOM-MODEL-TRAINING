Blind test-set audit (2026-10-01), used ONLY to compare models fairly. The real test labels were not changed.

v3 scored lower than v2 on the frozen test labels (0.368 vs 0.385 mAP50), and it had more "false alarms".
To check whether those were real lesions, every v2 or v3 detection on test with conf >= 0.3 and no label
(149 in total, duplicates merged) was put on sheets WITHOUT model name or confidence, then reviewed with the
round-3 rules. 82 were real lesions -> configs/label_fixes_test_audit.csv (test 889 -> 971 boxes).

Scores on the audited test labels (scripts/evaluate.py, imgsz 800):
                     round1 (-2)   v2      v3
  mAP@0.5              0.402      0.429   0.426
  mAP@0.5:0.95         0.216      0.232   0.235
  best F1              0.484      0.502   0.500
  FP/img @ conf 0.5    0.206      0.376   0.488
v2 and v3 are tied. Round 1 is slightly disadvantaged: its own unlabelled detections were not audited.
