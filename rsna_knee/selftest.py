"""
selftest.py - runs the whole training contract on synthetic data.

No competition data, no GPU, no network required (pretrained=False).  Run this
after editing any knee_*.py; it catches shape and masking bugs in seconds
instead of six GPU hours into step 4.

    python selftest.py
"""
import math
import os
import shutil
import tempfile

import numpy as np
import pandas as pd
import torch

import knee_common as kc
from knee_data import KneeStudyDataset, collate, make_folds
from knee_model import CareNet, SoftAsymmetricLoss, ModelEMA, MEDIAL_IDX, LATERAL_IDX

N_STUDIES = 8


def build_fake(tmp: str, cfg: kc.Cfg):
    rows = []
    rng = np.random.default_rng(0)
    for s in range(N_STUDIES):
        study = f"1.2.3.{s}"
        # deliberately uneven protocols: study 3 has no axial, study 5 only one series
        combos = [("Sagittal", 1, 1), ("Coronal", 1, 1), ("Axial", 1, 0), ("Sagittal", 0, 0)]
        if s == 3:
            combos = [c for c in combos if c[0] != "Axial"]
        if s == 5:
            combos = combos[:1]
        for j, (plane, fluid, fat) in enumerate(combos):
            series = f"{study}.{j}"
            vol = rng.integers(0, 255, (cfg.n_slices, cfg.img_size, cfg.img_size), dtype=np.uint8)
            kc.write_sprite(kc.sprite_path(tmp, study, series), vol, cfg.grid_w, cfg.jpeg_quality)
            rows.append(dict(StudyInstanceUID=study, SeriesInstanceUID=series,
                             Anatomical_Plane=plane, Fluid_Sensitive=fluid,
                             Fat_Suppression=fat, Laterality=int(s % 3)))
    series_df = pd.DataFrame(rows)
    series_df = pd.concat([kc.assign_slots(g) for _, g in series_df.groupby("StudyInstanceUID")])
    studies = pd.DataFrame({"StudyInstanceUID": [f"1.2.3.{s}" for s in range(N_STUDIES)]})
    y = rng.integers(0, 2, (N_STUDIES, kc.N_LABELS)).astype(np.float32)
    y[:, 0] = [0, 1] * (N_STUDIES // 2)          # keep at least one label balanced
    for i, c in enumerate(kc.LABELS):
        studies[c] = y[:, i]
    return studies, series_df.reset_index(drop=True), y


LEXICON_CASES = [
    # The second block is VERBATIM text from the competition corpus, pulled
    # from step 1 cell 5. Invented cases test what I imagined a report says;
    # these test what they actually say - which is how the Bulgarian, Greek,
    # Croatian and Turkish 'medyal' gaps were found in the first place.
    ('Complete tear of the anterior cruciate ligament. Small joint effusion.',
     {'ACL', 'Effusion'}, {'Fracture', 'MCL'}),
    ('ACL intact. Menisci intact. No fracture. Unremarkable study.',
     set(), {'Fracture', 'ACL', 'Medial Meniscus'}),
    ('Normal ACL. Normal MCL. Grade 2 signal in the medial meniscus.',
     {'Medial Meniscus'}, {'ACL', 'MCL'}),
    ('The medial meniscus shows no tear.',
     set(), {'Medial Meniscus'}),
    ('Medial meniscus: posterior horn tear. Lateral meniscus: intact.',
     {'Medial Meniscus'}, {'Lateral Meniscus'}),
    ('Tear of the medial meniscus, intact lateral meniscus.',
     {'Medial Meniscus'}, {'Lateral Meniscus'}),
    ('Impaction fracture of the lateral femoral condyle with bone marrow edema.',
     {'Fracture', 'Contusion'}, set()),
    ('Severe patellofemoral osteoarthritis with cartilage loss. Medial compartment osteoarthritis.',
     {'PF OA', 'Medial OA'}, {'Lateral OA'}),
    ('Baker cyst measuring 25 mm. No other abnormality.',
     {"Baker's"}, {'Lateral Meniscus', 'Medial Meniscus'}),
    ('Joint effusion, 12 mm deep.',
     {'Effusion'}, {'Lateral Meniscus', 'Medial Meniscus'}),
    ('Kein Erguss. Riss des Innenbandes.',
     {'MCL'}, {'Effusion'}),
    ('Bakerzyste in der Kniekehle. Knochenmarkodem medial tibial.',
     {"Baker's", 'Contusion'}, {'Fracture'}),
    ('Rottura del legamento crociato anteriore; versamento articolare.',
     {'ACL', 'Effusion'}, {'MCL'}),
    ('Lesao do menisco medial grau III. Cisto de Baker.',
     {"Baker's", 'Medial Meniscus'}, set()),
    ('Le ligament croise anterieur est intact. Dechirure du menisque lateral.',
     {'Lateral Meniscus'}, {'ACL'}),
    ('лезия на заден рог на медиален мениск, като лезията достига артикуларната повърност.',
     {'Medial Meniscus'}, {'Lateral Meniscus'}),
    ('няма мр данни за ставен излив. виждат се дифузни зони на повишен сигнален интензитет съответстващи на костномозъчен едем.',
     {'Contusion'}, {'Effusion'}),
    ('мр данни за ставен излив. предна кръстна връзка е руптурирана и не се проследява до залавните си места.',
     {'ACL', 'Effusion'}, set()),
    ('медиален и латерален менискус – с нормален мр образ.',
     set(), {'Lateral Meniscus', 'Medial Meniscus'}),
    ('начални дегенеративни промени по латералния мениск.',
     {'Lateral Meniscus'}, {'Medial Meniscus'}),
    ('двата кръстни лигамента се проследяват до залавните си места с правилна форма.',
     set(), {'ACL'}),
    ('medyal meniskus arka boynuzdan govdesine uzanan longitudinal yirtik.',
     {'Medial Meniscus'}, {'Lateral Meniscus'}),
    ('eklemde sivi artisi saptanmamistir.',
     set(), {'Effusion'}),
    ('diz eklemi ici sivi miktari hafif derecede artmis.',
     {'Effusion'}, set()),
    ('lateral meniskuste grade ii dejenerasyon, medial meniskuste grade iii dejenerasyon izlenmistir.',
     {'Lateral Meniscus', 'Medial Meniscus'}, set()),
    ('medyal ve lateral meniskus normal. eklem kikirdaklari ve kemikler normal.',
     set(), {'Lateral Meniscus', 'Medial Meniscus'}),
    ('arka capraz ve yan baglar korunmus. on capraz bagda tam kata yakin yirtik izleniyor.',
     {'ACL'}, set()),
    ('medijalni menisk bez znakova degeneracije ili rupture.',
     set(), {'Medial Meniscus'}),
    ('prednji krizni ligament urednog signala te se prati u kontinuitetu.',
     set(), {'ACL'}),
    ('plitke fisure hrskavice medijalne fasete patele (2. stupanj hondromalacije).',
     {'PF OA'}, set()),
    ('δεν αναγνωριστηκαν παθολογικα ευρηματα απο τον ελεγχο των μηνισκων, των χιαστων, των πλαγιων συνδεσμων.',
     set(), {'ACL'}),
    ('χωρις ενδαρθρικη συλλογη υγρου.',
     set(), {'Effusion'}),
    ('rotura de menisco interno. condropatia femorotibial medial. derrame.',
     {'Effusion', 'Medial Meniscus', 'Medial OA'}, set()),
    ('ligamentos cruzados y colaterales dentro de limites normales.',
     set(), {'ACL'}),
    ('gevorderd lateraal femorotibiaal kraakbeenlijden met volledig kraakbeenverlies anterieur en van het laterale tibiaplateau',
     {'Lateral OA'}, {'Medial OA'}),
    ('ulceras condrales de espesor total de la region central y de carga del condilo femoral medial',
     {'Medial OA'}, {'Lateral OA'}),
    ('small joint effusion with synovial thickening compatible with synovitis. there is popliteal cyst measuring 21 x 17 x 35 mm.',
     {'Synovitis', 'Effusion', "Baker's"}, set()),
    ('kostani edem medijalnog kondila femura.',
     {'Contusion'}, set()),
    ('erozivne promjene zglobne hrskavice medijalnog kompartmenta femorotibijalnog zgloba.',
     {'Medial OA'}, {'Lateral OA'}),
    ('geen vocht in het gewricht. voorste kruisband ongestoord.',
     set(), {'ACL', 'Effusion'}),
    ('mr knie rechts. bevindingen: ruptuur van de voorste kruisband.',
     {'ACL'}, set()),
    ('zglobna hrskavica medijalnog kompartmenta uredna.',
     set(), {'Medial OA'}),
    ('hondromalacija patele 4. stupanj, potpuni gubitak hrskavice.',
     {'PF OA'}, set()),
    ('patellar kondromalazi 3. derece.',
     {'PF OA'}, set()),
    ('chondral thinning of the patella, grade 3.',
     {'PF OA'}, set()),
    ('distal kuadriseps ve patellar tendonlar normaldir.',
     set(), {'PF OA'}),
    ('the patellar cartilage is normal.',
     set(), {'PF OA'}),
]


def test_amp_compat():
    """Kaggle is on torch 2.10 and moving; the AMP spelling this code uses must
    resolve on whatever version the notebook happens to get."""
    with kc.amp_autocast("cpu"):
        y = torch.randn(4, 4) @ torch.randn(4, 4)
    assert y.shape == (4, 4)
    scaler = kc.make_grad_scaler("cpu")
    assert hasattr(scaler, "scale") and hasattr(scaler, "step")
    print(f"[12] AMP compat on torch {torch.__version__}: "
          f"autocast -> {y.dtype}, scaler -> {type(scaler).__name__}")


def test_path_discovery():
    """The competition slug and mount layout have both moved before; nothing in
    the pipeline may hard-code them."""
    root = tempfile.mkdtemp(prefix="kaggleinput_")
    try:
        decoy = os.path.join(root, "someones-dataset")
        os.makedirs(decoy)
        open(os.path.join(decoy, "sample_submission.csv"), "w").close()

        real = os.path.join(root, "competitions", "rsna-knee-abnormality-detection")
        os.makedirs(os.path.join(real, "train_series"))
        for f in ("sample_submission.csv", "train_series.csv", "test_series.csv"):
            open(os.path.join(real, f), "w").close()

        found = kc.find_comp_dir(root=root)
        assert found == real, f"picked the decoy: {found}"
        assert kc.find_series_root(found, "train") == os.path.join(real, "train_series")
        try:
            kc.find_comp_dir(root=os.path.join(root, "nothing-here"))
            raise AssertionError("missing data should raise")
        except FileNotFoundError:
            pass
        print("[13] competition path autodetected past a decoy; missing data raises clearly")
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_lexicon():
    """The report rules decide the teacher's floor - a leaked negation here
    poisons every soft label downstream, so they get real cases."""
    from knee_text import (norm_text, rule_features, ANATOMY, ABNORMAL,
                           OA_TERMS, NEGATION, NEGATION_AFTER)
    # NFKD decomposes Cyrillic and Turkish letters - й becomes и, ё becomes е -
    # so a pattern written in the natural spelling matches nothing at all, with
    # no error to notice. Every pattern must be in post-normalisation form.
    pats = dict(ANATOMY)
    pats.update({"ABNORMAL": ABNORMAL, "OA_TERMS": OA_TERMS,
                 "NEGATION": NEGATION, "NEGATION_AFTER": NEGATION_AFTER})
    unnormalised = [(n, c) for n, p in pats.items() for c in set(p)
                    if c.isalpha() and norm_text(c) != c]
    assert not unnormalised, f"patterns hold characters norm_text alters: {unnormalised}"

    # Grading check: the same finding must rank by severity, since the rules
    # feed an AUC-scored pipeline and a flat binary cannot separate a trace
    # effusion from a large one.
    eff = kc.LABELS.index("Effusion")
    grades = [rule_features(norm_text(t))[eff] for t in
              ("Large joint effusion.", "Joint effusion.", "Minimal joint effusion.",
               "No joint effusion.")]
    assert grades[0] > grades[1] > grades[2] > grades[3] == 0, grades

    # Radiology grades, which carry far more signal than adjectives: grade I-II
    # meniscal signal is intrasubstance degeneration, grade III is a tear.
    mm = kc.LABELS.index("Medial Meniscus")
    g2 = rule_features(norm_text("Grade 2 signal in the medial meniscus."))[mm]
    g3 = rule_features(norm_text("Grade 3 tear of the medial meniscus."))[mm]
    assert g3 > g2 > 0, (g2, g3)
    print(f"[13b] severity grading: large {grades[0]:.2f} > plain {grades[1]:.2f} "
          f"> minimal {grades[2]:.2f} > negated {grades[3]:.2f}; "
          f"meniscus grade III {g3:.2f} > grade II {g2:.2f}")

    bad = []
    for txt, must, mustnot in LEXICON_CASES:
        f = rule_features(norm_text(txt))
        hits = {kc.LABELS[i] for i in range(kc.N_LABELS) if f[i] > 0}
        if not (must <= hits) or (mustnot & hits):
            bad.append((txt, sorted(must - hits), sorted(mustnot & hits)))
    for t, miss, extra in bad:
        print(f"    LEXICON FAIL {t[:60]!r} missing={miss} false={extra}")
    assert not bad, f"{len(bad)}/{len(LEXICON_CASES)} lexicon cases failed"
    # Knee side from the report: the DICOM carries none for ~48% of studies.
    from knee_text import laterality_from_report as _lat
    side_cases = [
        ("мр находка: дясната колянна става", 1),
        ("мр: лявата колянна става", 0),
        ("μαγνητικη τομογραφια ∆εξιου γονατος", 1),
        ("sol diz mrg. tetkik protokolu", 0),
        ("sag diz mrg. bulgular", 1),
        ("mr knie rechts 15ch aa", 1),
        ("mri of left knee with -locator", 0),
        ("findings: small joint effusion", 2),
    ]
    for _txt, _want in side_cases:
        _got = _lat(norm_text(_txt))
        assert _got == _want, f"side {_got} != {_want} for {_txt[:40]!r}"
    # U+2206 INCREMENT is a maths symbol, not capital delta. One site writes
    # its Greek reports with it, so "not" reads as ∆εν and every Greek negation
    # was missed until norm_text mapped it.
    assert norm_text("∆εν") == "δεν", "INCREMENT was not mapped to delta"
    print(f"[13c] knee side from report: {len(side_cases)}/{len(side_cases)} cases; "
          f"INCREMENT delta normalised")


    print(f"[14] report lexicon: {len(LEXICON_CASES)}/{len(LEXICON_CASES)} cases pass "
          f"(en/es/pt/fr/de/it/tr/bg/el/hr); no unnormalised pattern characters")


def test_train_loop(studies, series_df, y, cfg, tmp):
    """Two optimiser steps end to end: param groups, accumulation, EMA eval,
    checkpoint save and reload through the same path step 5 uses."""
    import torch.utils.data as tud
    ds = KneeStudyDataset(studies, series_df, cfg, train=True, targets=y)
    dl = tud.DataLoader(ds, batch_size=2, shuffle=True, collate_fn=collate, num_workers=0)
    model = CareNet(cfg)
    lossf = SoftAsymmetricLoss()
    ema = ModelEMA(model, 0.9)

    bb = [p for n, p in model.named_parameters() if n.startswith("backbone.")]
    hd = [p for n, p in model.named_parameters() if not n.startswith("backbone.")]
    assert bb and hd, "param grouping found an empty group"
    opt = torch.optim.AdamW([{"params": bb, "lr": 1e-4}, {"params": hd, "lr": 5e-4}])

    before = model.head.base.weight.detach().clone()
    for i, b in enumerate(dl):
        o = model(b)
        loss = lossf(o["logits"], b["target"], b["weight"])
        loss = loss + 0.3 * (lossf(o["logit_med"], b["target"][:, MEDIAL_IDX], b["weight"]) +
                             lossf(o["logit_lat"], b["target"][:, LATERAL_IDX], b["weight"]))
        loss.backward()
        opt.step(); opt.zero_grad(set_to_none=True); ema.update(model)
        if i >= 1:
            break
    assert not torch.allclose(before, model.head.base.weight), "weights did not move"
    print(f"[15] two optimiser steps ran; head weight moved by "
          f"{(model.head.base.weight - before).abs().mean().item():.2e}")

    # eval path must produce one row per study and a finite AUC
    dl_va = tud.DataLoader(KneeStudyDataset(studies, series_df, cfg, train=False, targets=y),
                           batch_size=2, collate_fn=collate, num_workers=0)
    P, Y = [], []
    ema.ema.eval()
    with torch.no_grad():
        for b in dl_va:
            P.append(torch.sigmoid(ema.ema(b)["logits"]).numpy())
            Y.append(b["target"].numpy())
    P, Y = np.concatenate(P), np.concatenate(Y)
    assert P.shape == (len(studies), kc.N_LABELS), P.shape
    m, _ = kc.macro_auc(Y, P)
    assert np.isfinite(m)
    print(f"[16] eval produced {P.shape} predictions, macro AUC {m:.3f}")

    # checkpoint round trip exactly as step 5 does it
    ckpt = os.path.join(tmp, "ck.pt")
    torch.save({"model": ema.ema.state_dict(), "cfg": cfg.__dict__, "fold": 0, "auc": m}, ckpt)
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    cfg2 = kc.Cfg(**{k: v for k, v in ck["cfg"].items() if k in kc.Cfg.__dataclass_fields__})
    cfg2.pretrained = False
    m2 = CareNet(cfg2)
    m2.load_state_dict(ck["model"])
    m2.eval()
    with torch.no_grad():
        b = next(iter(dl_va))
        a = torch.sigmoid(ema.ema(b)["logits"])
        c = torch.sigmoid(m2(b)["logits"])
    assert torch.allclose(a, c, atol=1e-5), (a - c).abs().max().item()
    print("[17] checkpoint save/reload reproduces identical predictions")


def main():
    tmp = tempfile.mkdtemp(prefix="kneetest_")
    try:
        cfg = kc.Cfg(img_size=64, n_slices=4, grid_w=2, cache_dir=tmp,
                     backbone="resnet18", pretrained=False, embed_dim=64,
                     slice_layers=1, study_layers=1, n_heads=4, batch_size=2)
        kc.seed_everything(0)
        studies, series_df, y = build_fake(tmp, cfg)
        print(f"[1] fake cache: {len(series_df)} series / {N_STUDIES} studies")
        print("    slot assignment:\n",
              series_df.groupby(["Anatomical_Plane", "Fluid_Sensitive"])["slot"].apply(list).to_string())

        ds = KneeStudyDataset(studies, series_df, cfg, train=True, targets=y)
        item = ds[0]
        assert item["image"].shape == (kc.N_SLOTS, cfg.n_slices, 3, cfg.img_size, cfg.img_size), item["image"].shape
        print(f"[2] item image {tuple(item['image'].shape)} mask {item['series_mask'].tolist()} "
              f"lat {int(item['lat'])}")

        # a study missing a plane must mask that slot, not crash
        i3 = ds.uids.index("1.2.3.3")
        assert int(ds[i3]["series_mask"].sum()) < kc.N_SLOTS
        i5 = ds.uids.index("1.2.3.5")
        assert int(ds[i5]["series_mask"].sum()) == 1
        print(f"[3] missing-sequence studies masked correctly "
              f"({int(ds[i3]['series_mask'].sum())} and {int(ds[i5]['series_mask'].sum())} of {kc.N_SLOTS} slots)")

        dl = torch.utils.data.DataLoader(ds, batch_size=cfg.batch_size, shuffle=True,
                                         collate_fn=collate, num_workers=0)
        batch = next(iter(dl))
        model = CareNet(cfg)
        n_par = sum(p.numel() for p in model.parameters()) / 1e6
        out = model(batch)
        assert out["logits"].shape == (cfg.batch_size, kc.N_LABELS), out["logits"].shape
        assert out["logit_med"].shape == (cfg.batch_size, len(MEDIAL_IDX))
        assert out["logit_lat"].shape == (cfg.batch_size, len(LATERAL_IDX))
        print(f"[4] CareNet {n_par:.1f}M params -> logits {tuple(out['logits'].shape)}, "
              f"med {tuple(out['logit_med'].shape)}, lat {tuple(out['logit_lat'].shape)}")

        lossf = SoftAsymmetricLoss()
        loss = lossf(out["logits"], batch["target"], batch["weight"])
        loss.backward()
        grads = [p.grad.abs().sum().item() for p in model.parameters() if p.grad is not None]
        assert np.isfinite(loss.item()) and sum(grads) > 0
        print(f"[5] loss {loss.item():.4f}, {len(grads)} tensors received gradient")

        # soft targets must be accepted, not just 0/1
        soft = torch.rand_like(batch["target"])
        assert np.isfinite(lossf(out["logits"].detach(), soft).item())
        print("[6] soft targets accepted by the loss")

        ema = ModelEMA(model, 0.9)
        ema.update(model)
        print("[7] EMA update ok")

        # mirror TTA must keep the shape and flip the laterality token cleanly
        flipped = dict(batch)
        flipped["image"] = torch.flip(batch["image"], dims=[-1])
        flipped["lat"] = torch.where(batch["lat"] < 2, 1 - batch["lat"], batch["lat"])
        with torch.no_grad():
            o2 = model(flipped)
        assert o2["logits"].shape == out["logits"].shape
        # The compartment branch must switch OFF when the side is unknown: with
        # ~48% of studies carrying no laterality, a coin-flip half-to-compartment
        # mapping trains the medial head on lateral anatomy.
        model.eval()          # dropout would make two forward passes differ
        b_known = dict(batch); b_known["lat"] = torch.zeros_like(batch["lat"])
        b_unk = dict(batch); b_unk["lat"] = torch.full_like(batch["lat"], 2)
        with torch.no_grad():
            o_known, o_unk = model(b_known), model(b_unk)
        assert o_known["comp_valid"].sum() == len(batch["lat"])
        assert o_unk["comp_valid"].sum() == 0
        # with the side unknown the compartment logits must not reach the output
        with torch.no_grad():
            base = model(b_unk)["logits"][:, MEDIAL_IDX]
            model.comp_mix.data.fill_(5.0)        # force the mix wide open
            still = model(b_unk)["logits"][:, MEDIAL_IDX]
            model.comp_mix.data.fill_(0.0)
        assert torch.allclose(base, still, atol=1e-5), \
            "compartment branch still influences an unknown-side study"
        model.train()
        print("[7b] compartment branch is inert when laterality is unknown")

        print(f"[8] mirror TTA ok (lat {batch['lat'].tolist()} -> {flipped['lat'].tolist()})")

        # sprite round trip must be lossless in shape and close in value
        vol = np.tile(np.arange(cfg.img_size, dtype=np.uint8), (cfg.n_slices, cfg.img_size, 1))
        p = os.path.join(tmp, "rt.jpg")
        kc.write_sprite(p, vol, cfg.grid_w, 100)
        back = kc.read_sprite(p, cfg.n_slices, cfg.img_size, cfg.grid_w)
        assert back.shape == vol.shape
        print(f"[9] sprite round-trip {back.shape}, mean abs err {np.abs(back.astype(int)-vol.astype(int)).mean():.2f}")

        # A series where NOTHING decodes must be rejected, not returned black.
        # Each failed slice is replaced by its neighbour, which is right for one
        # bad slice and catastrophic for a whole series: a missing transfer-
        # syntax plugin would otherwise fill the cache with black sprites,
        # report zero failures, and train the model on nothing.
        bad = os.path.join(tmp, "undecodable", "series")
        os.makedirs(bad, exist_ok=True)
        for i in range(4):
            with open(os.path.join(bad, f"{i}.dcm"), "wb") as fh:
                fh.write(b"not a dicom at all")
        assert kc.load_series_volume(bad, cfg.n_slices, cfg.img_size) is None, \
            "an undecodable series came back as a volume"
        print("[9b] a series that decodes no slices returns None, not black pixels")

        # metric + rank normalisation
        pred = np.random.rand(50, kc.N_LABELS)
        truth = (np.random.rand(50, kc.N_LABELS) > 0.7).astype(np.float32)
        m, per = kc.macro_auc(truth, pred)
        rn = kc.rank_normalise(pred)
        m2, _ = kc.macro_auc(truth, rn)
        assert abs(m - m2) < 1e-9, (m, m2)
        print(f"[10] macro AUC {m:.4f}; rank-normalisation is order-preserving (delta {abs(m-m2):.2e})")

        # Sparse gold labels: only annotated cells may be scored, and a column
        # with too few annotated rows must report nan, not a fabricated number.
        mask = np.zeros_like(truth)
        mask[:, 0] = 1
        mask[:3, 1] = 1
        _, per_m = kc.macro_auc(truth, pred, mask=mask)
        assert not math.isnan(per_m[kc.LABELS[0]]), "fully annotated column skipped"
        assert math.isnan(per_m[kc.LABELS[2]]), "unannotated column scored anyway"
        scored = sum(1 for v in per_m.values() if not math.isnan(v))
        print(f"[10b] masked metric scores {scored} annotated column(s), nan for the rest")

        # Soft y_true: the rules are graded and the teacher emits probabilities,
        # so both callers pass continuous targets. roc_auc_score rejects those
        # outright - "continuous format is not supported" - which is a crash
        # mid-epoch, not a wrong number, and step 4 validates against teacher
        # probabilities too.
        soft = np.random.choice([0.0, 0.35, 0.7, 1.0], size=truth.shape)
        m_soft, _ = kc.macro_auc(soft, pred)
        m_hard, _ = kc.macro_auc((soft > 0.5).astype(np.float64), pred)
        assert abs(m_soft - m_hard) < 1e-12, (m_soft, m_hard)
        assert np.isfinite(kc.macro_auc(np.random.rand(*truth.shape), pred)[0])
        print(f"[10c] soft targets binarise at 0.5: {m_soft:.4f} == explicit {m_hard:.4f}")

        folds = make_folds(studies, 4, seed=0)
        assert len(set(folds.tolist())) > 1
        print(f"[11] folds {folds.tolist()}")

        test_amp_compat()
        test_path_discovery()
        test_lexicon()
        test_train_loop(studies, series_df, y, cfg, tmp)
        print("\nALL SELF-TESTS PASSED")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
