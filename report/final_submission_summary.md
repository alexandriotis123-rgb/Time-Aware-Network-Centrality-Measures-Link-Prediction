**Executive Summary**

Αυτό το έγγραφο συνοψίζει τις αλλαγές που εφαρμόστηκαν στον κώδικα της εργασίας "Temporal Network Analysis" και περιγράφει τη νέα πειραματική ροή, τα κριτήρια αξιολόγησης, και τα αποτελέσματα από ένα debug-run. Όλες οι αλλαγές έγιναν για να διασφαλιστεί σωστή εκπαίδευση σε temporal link-prediction, αναπαραγωγιμότητα και υπεράσπιση της μεθοδολογίας.

**Κύρια Αποτελέσματα**

- **Holdout test set:** Προστέθηκε σταθερό holdout (`TEST_HOLDOUT_RATIO`) για κάθε σετ υποψηφίων. Η υπόλοιπη συλλογή χρησιμοποιείται για train/validation split.
- **Σωστά labels:** Τα labels πλέον είναι "νέα links" — δηλαδή `E2 \ E1` (εμφάνιση στο graph_2 αλλά όχι στο graph_1).
- **Canonicalization edges:** Όλα τα υποψήφια άκρα ομαδοποιούνται ως μη-κατευθυντικά με canonical μορφή `(min(u,v), max(u,v))` ώστε να αποφεύγονται διπλές αναπαραστάσεις.
- **Persistent nodes invariant:** Οι υποψήφιες ακμές σχηματίζονται μόνο ανάμεσα σε κόμβους που είναι persistent (παρόντες και στα δύο χρονικά γράφηματα) — υλοποιημένο σε [src/preprocessing/persistent_nodes.py](src/preprocessing/persistent_nodes.py) και [src/preprocessing/candidate_edges.py](src/preprocessing/candidate_edges.py).
- **Σταθερό, εξηγήσιμο downsampling ομοιογενών σκορ:** Για να περιοριστεί το search space των πιθανών similarity thresholds, τα μοναδικά similarity scores downsampled σε `MAX_UNIQUE_SCORES` αντιπροσωπευτικές τιμές που είναι ομοιόμορφα κατανεμημένες κατά μήκος της σειράς των μοναδικών τιμών (evenly-spaced sampling). Αυτό παρέχει απλό, εξηγήσιμο σχήμα δειγματοληψίας αντί για τυχαία ή κλασματική κόψιμο.
- **Multi-interval (union) training + rollback:** Η διαδικασία βελτιστοποίησης `improve_range_set` προσθέτει μη-επικαλυπτόμενα intervals μόνο εάν προσφέρουν τουλάχιστον `MIN_IMPROVEMENT` στην accuracy, αποφεύγοντας θορυβικές μικροβελτιώσεις.
- **Διόρθωση shortest-path similarity:** Οι αποσυνδεδεμένες ζεύξεις αντιμετωπίζονται με -inf ώστε να μην προκαλούν ψευδείς υψηλές ομοιότητες.
- **Evaluation fixes:** Το μέτρο ακρίβειας/TPR/TNR υπολογίζεται πάντα πάνω στο σύνολο των υποψηφίων (candidate population). Οι μετρικές δεν επιβαρύνονται από αρχεία εδάφους εκτός των υποψηφίων.

**Αλλαγμένα/Προστιθέμενα αρχεία (κύρια)**

- [config.py](config.py): Προσθήκη `TEST_HOLDOUT_RATIO`, `DEBUG` flags και ρυθμίσεις σχετικές με `MAX_UNIQUE_SCORES`.
- [src/preprocessing/persistent_nodes.py](src/preprocessing/persistent_nodes.py): `find_persistent_nodes`, `restrict_graph`, `build_persistent_pairs`.
- [src/preprocessing/candidate_edges.py](src/preprocessing/candidate_edges.py): `build_candidate_edges` με canonicalization και exclusion των ήδη υπαρχόντων άκρων.
- [src/preprocessing/feature_vectors.py](src/preprocessing/feature_vectors.py): ευθυγράμμιση χαρακτηριστικών από `graph_1`.
- [src/preprocessing/labels.py](src/preprocessing/labels.py): labels = `graph_2 \ graph_1` (νέα links).
- [src/prediction/training.py](src/prediction/training.py): representative downsampling, `MIN_IMPROVEMENT`, επιστροφή `candidate_ranges`, `train_similarity_measure` και `improve_range_set` συμπεριφορές.
- [src/prediction/experiment.py](src/prediction/experiment.py): holdout test split + train/validation split, βελτιστοποίηση με validation και τελική αξιολόγηση στο test.
- [src/prediction/evaluation.py](src/prediction/evaluation.py): `_compute_accuracy_from_dataset` περιορίζει υπολογισμούς στο σύνολο των υποψηφίων.

**Νέα/βελτιωμένα σημαντικά σημεία στην ροή εκτέλεσης**

1. Επιλογή persistent node pairs (μόνο κόμβοι κοινών σε G1,G2).
2. Δημιουργία υποψήφιων ακμών ανάμεσα στους persistent κόμβους με canonical μορφή.
3. Διαχωρισμός υποψηφίων: σταθερό holdout test (`TEST_HOLDOUT_RATIO`), και το υπόλοιπο για balanced train+validation pool.
4. Split του pool σε `train` και `validation` (π.χ. validation ratio 0.2).
5. Train: brute-force search πάνω σε candidate thresholds (με representative sampling των unique scores) για την εύρεση αρχικού best-range.
6. Improve: `improve_range_set` προσθέτει επιπλέον μη-επικαλυπτόμενα interval εάν η βελτίωση > `MIN_IMPROVEMENT` με έλεγχο σε validation.
7. Τελική αξιολόγηση στο holdout `test` set και export των μετρικών.

**Ορισμός Accuracy & Διευκρινίσεις**

- Η accuracy υπολογίζεται ως σταθμισμένος συνδυασμός TPR και TNR με βάρος λ = |E_true_in_candidates| / |E_candidates| (αναπαριστά το class imbalance εντός του χώρου υποψηφίων).
- Όλες οι μετρικές (Precision, Recall, P@K, TPR, TNR) υπολογίζονται αποκλειστικά πάνω στο σύνολο των `candidate_edges` που ελέγχονται.

**Σύντομο απόσπασμα από debug-run (συνοπτικά)**

- Persistent pairs: 9
- Candidate edges per pair (debug cap): 200,000
- Test holdout: 40,000 (20%)
- Balanced train+val pool (παράδειγμα Pair 1): 1,124 → Train:900 Val:224
- Συχνά το `PA` έδωσε τις καλύτερες ACC/REC σε πολλά persistent pairs στο debug-run· ωστόσο αυτό ποικίλλει ανά pair.

Παράδειγμα μετρικών (Pair 1, συνοπτικά):

- Train ACC (PA): ~0.6644, TPR~0.7004, TNR~0.6278
- Test sizes: 40,000 (holdout)

Πλήρες raw log εξαγόμενο από το run καταγράφεται στο session log (τοπικός αρχείο εκτέλεσης). Για άμεση επανάληψη του ίδιου run:

```bash
python main.py
```

Για να τρέξεις τα tests:

```bash
pytest -q
```

**Σημειώσεις για reproducibility και επιλογές σχεδιασμού**

- Η downsampling πολιτική (evenly-spaced unique-score sampling) επιλέχθηκε επειδή είναι αναπαραγώγιμη και εύκολη να δικαιολογηθεί σε έκθεση: επιλέγουμε αντιπροσωπευτικές τιμές που καλύπτουν ολόκληρο το φάσμα, αντί να κόβουμε τα πρώτα N ή να παίρνουμε τυχαία δείγματα.
- `MIN_IMPROVEMENT` αποτρέπει την αποδοχή μη-σημαντικών αριθμητικών βελτιώσεων.
- Canonicalization και persistent-node invariant εμποδίζουν διαρροή labels και διπλές εγγραφές.

**Προτεινόμενα επόμενα βήματα**

- (Προαιρετικό) Τρέξε full-scale (χωρίς debug caps) για οριστικά αποτελέσματα και σχήματα — θα απαιτήσει σημαντικούς πόρους.
- Επέκταση με περισσότερες μετρικές (AUC-ROC, NDCG) αν χρειαστεί συγκριτική αξιολόγηση.
- Οπτικοποίηση ROC/Precision-Recall ανά similarity μέτρο και ανά persistent pair για καλύτερη παρουσίαση.

**Συνοπτικό log αλλαγών (commit-level)**

- Εφαρμογή: labels -> `graph_2 \ graph_1` (fix label leakage).
- Εφαρμογή: canonicalization των ακμών στην κατασκευή υποψηφίων.
- Εφαρμογή: holdout test + train/validation split στο `experiment.py`.
- Εφαρμογή: representative downsampling των unique similarity scores σε `training.py`.
- Εφαρμογή: `improve_range_set` με `MIN_IMPROVEMENT`.
- Εφαρμογή: evaluation περιορισμένη στο σύνολο υποψηφίων.

---

Αν θες, μπορώ να:

- Προσθέσω τα full run logs μέσα σε αυτό το αρχείο ή σε [report/](report/) ως ξεχωριστό αρχείο.
- Παραγάγω εικόνες μετρικών για κάθε persistent pair και να τις βάλω στο [outputs/figures/](outputs/figures/).

Πες μου ποιο από τα παραπάνω προτιμάς και το προχωρώ.