Label review, round 1 (2026-09-30)

Model yolo26s_stemcanker-2/best.pt was run on every image. Each detection with conf >= 0.3 that
touched no existing label (IoU < 0.1, not inside a label, no label inside it) became a candidate:
867 in total (train 617, val 150, test 100).

sheet_XX.jpg   crops of the candidates. Yellow number = candidate id (candidates.csv).
               RED box = candidate, GREEN boxes = labels that were already there.
accepted.txt   ids judged to be real lesions (612) -> configs/label_fixes_v2.csv
               -> added by scripts/apply_label_fixes.py

Rejected or skipped (255): areoles/spines, insects, debris/threads, glare, black padding edges,
boxes that span two lesions, and faint white spots that could be early lesions but could not be
told apart from scale insects/scars with confidence. If you know those white spots ARE stem canker,
add them to configs/label_fixes_v2.csv (convert x1 y1 x2 y2 from candidates.csv
to xc yc w h) and rerun scripts/apply_label_fixes.py.

Not fixed by this round: lesions the model never detected at all (e.g. the large grey necrotic
cankers in photo 169). Run the same review with the v2 model to catch more of them.
