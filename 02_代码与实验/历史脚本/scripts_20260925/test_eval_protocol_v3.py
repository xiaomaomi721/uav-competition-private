"""Unit tests for EVAL_PROTOCOL_V3."""
import json, sys
import numpy as np
sys.path.insert(0, ".")
from eval_protocol_v3 import ConfusionMatrix, official_to_train, evaluate_pair, NUM_CLASSES

results = []

def test(name, fn):
    try:
        fn()
        results.append({"test": name, "status": "PASS"})
    except Exception as e:
        results.append({"test": name, "status": "FAIL", "error": str(e)})

# Test 1: GT with 255 doesn't pollute valid classes
def t1():
    cm = ConfusionMatrix()
    gt = np.array([[0, 1, 255], [255, 3, 7]], dtype=np.int64)
    pred = np.array([[0, 1, 0], [1, 3, 7]], dtype=np.int64)
    cm.update(gt, pred)
    r = cm.compute_iou()
    assert r["valid_pixels"] == 4, f"expected 4 valid, got {r['valid_pixels']}"
    assert r["ignore_pixels"] == 2
test("GT_255_ignored_correctly", t1)

# Test 2: Official mapping 1-8 -> 0-7, 0 -> 255
def t2():
    official = np.array([[0, 1, 2], [8, 5, 3]], dtype=np.int64)
    train = official_to_train(official)
    assert train[0,0] == 255
    assert train[0,1] == 0
    assert train[0,2] == 1
    assert train[1,0] == 7
    assert train[1,1] == 4
    assert train[1,2] == 2
test("official_label_mapping", t2)

# Test 3: Empty class IoU handling
def t3():
    cm = ConfusionMatrix()
    gt = np.array([0, 0, 1, 1], dtype=np.int64)
    pred = np.array([0, 0, 1, 1], dtype=np.int64)
    cm.update(gt, pred)
    r = cm.compute_iou()
    assert r["per_class_iou"][2] is None, f"class 2 should be absent"
    assert 2 in r["absent_classes"]
test("empty_class_iou_is_None", t3)

# Test 4: Batch accumulation equals single batch
def t4():
    gt1 = np.array([[0,1],[2,3]], dtype=np.int64)
    pred1 = np.array([[0,1],[1,3]], dtype=np.int64)
    gt2 = np.array([[4,5],[6,7]], dtype=np.int64)
    pred2 = np.array([[4,5],[6,7]], dtype=np.int64)
    cm_batch = ConfusionMatrix()
    cm_batch.update(gt1, pred1)
    cm_batch.update(gt2, pred2)
    cm_single = ConfusionMatrix()
    gt_all = np.vstack([gt1, gt2])
    pred_all = np.vstack([pred1, pred2])
    cm_single.update(gt_all, pred_all)
    assert np.array_equal(cm_batch.matrix, cm_single.matrix)
test("batch_accumulation_equals_single", t4)

# Test 5: All pixels ignored -> no crash
def t5():
    cm = ConfusionMatrix()
    gt = np.full((4,4), 255, dtype=np.int64)
    pred = np.zeros((4,4), dtype=np.int64)
    cm.update(gt, pred)
    r = cm.compute_iou()
    assert r["valid_pixels"] == 0
    assert r["ignore_pixels"] == 16
test("all_ignored_no_crash", t5)

# Test 6: Pred out of range fails closed
def t6():
    cm = ConfusionMatrix()
    gt = np.array([0, 1, 2], dtype=np.int64)
    pred = np.array([0, 1, 8], dtype=np.int64)
    try:
        cm.update(gt, pred)
        raise AssertionError("should have raised")
    except ValueError:
        pass
test("pred_out_of_range_fails_closed", t6)

# Test 7: Merged confusion = correct combined mIoU
def t7():
    cm1 = ConfusionMatrix()
    cm1.update(np.array([0,1,2,3], dtype=np.int64), np.array([0,1,2,3], dtype=np.int64))
    cm2 = ConfusionMatrix()
    cm2.update(np.array([4,5,6,7], dtype=np.int64), np.array([4,5,6,7], dtype=np.int64))
    cm_merged = ConfusionMatrix()
    cm_merged.merge(cm1)
    cm_merged.merge(cm2)
    r = cm_merged.compute_iou()
    assert abs(r["mIoU"] - 1.0) < 1e-9
test("merged_confusion_mIoU", t7)

# Test 8: Vehicle (class 7) not polluted by ignore=255
def t8():
    cm = ConfusionMatrix()
    gt = np.array([[7, 255], [255, 7]], dtype=np.int64)
    pred = np.array([[7, 0], [1, 7]], dtype=np.int64)
    cm.update(gt, pred)
    r = cm.compute_iou()
    assert r["per_class_iou"][7] is not None
    assert r["per_class_iou"][7] > 0.9
test("vehicle_not_polluted_by_ignore", t8)

# Write results
out = {"tests": results, "all_pass": all(r["status"]=="PASS" for r in results)}
with open("TEST_RESULTS.json", "w") as f:
    json.dump(out, f, indent=2)
for r in results:
    print(f"{r['status']}: {r['test']}")
print(f"\nAll PASS: {out['all_pass']}")
if not out["all_pass"]:
    sys.exit(1)
