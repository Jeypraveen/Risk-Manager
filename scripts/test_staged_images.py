import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.vision.pipeline import run_vision_pipeline
from src.fusion.meta_learner import MetaLearner
from src.recovery.circuit_breaker import CircuitBreaker

pairs = [
    ('smartphone.png', 'smartphone_legitimate.png', False, 'Genuine smartphone'),
    ('smartphone.png', 'empty_box_fraud.png', True, 'Empty box smartphone'),
    ('smartphone.png', 'substitution_fraud.png', True, 'Substituted smartphone'),
    ('headphones.png', 'headphones_legitimate.png', False, 'Genuine headphones'),
    ('iphone.jpg', 'iphone_return.jpg', False, 'Genuine iPhone'),
    ('shoes_red.jpg', 'shoes_legitimate.jpg', False, 'Genuine shoes'),
    ('cetaphil.jpg', 'cetaphil_return.jpg', False, 'Genuine Cetaphil'),
    ('shoes_red.jpg', 'return_fraud_box_1787926845277.jpg', True, 'Substitution shoes'),
    ('bag.png', 'bag_nudge_blurry.jpg', False, 'Blurry bag (legit)'),
    ('iphone_17promax.jpg', 'iphone_return_legitimate.jpg', False, 'Genuine iPhone 17'),
]

def run():
    ml = MetaLearner()
    cb = CircuitBreaker()
    y_true = []
    y_pred = []
    print(f"{'Description':<30} | {'Sim':<5} | {'Empty':<5} | {'Tabular':<7} | {'Fusion':<6} | {'Label'}")
    print('-' * 75)
    for cat, ret, is_fraud, desc in pairs:
        cat_path = f'data/images/catalog/{cat}'
        ret_path = f'data/images/returns/{ret}'
        if not Path(cat_path).exists() or not Path(ret_path).exists():
            continue
            
        res = run_vision_pipeline(cat_path, ret_path, cb)
        sim = res.semantic_similarity
        empty = res.empty_box_flag
        conf = res.modality_confidence
        
        tab_score = 0.5
        trust_score = ml.predict(
            tabular_score=tab_score,
            semantic_similarity=sim,
            empty_box_flag=empty,
            modality_confidence=conf
        )
        pred_fraud_score = 1.0 - trust_score
        y_true.append(1 if is_fraud else 0)
        y_pred.append(pred_fraud_score)
        
        sim_str = f"{sim:.2f}" if sim is not None else "N/A"
        empty_str = "1" if empty else "0" if empty is not None else "N/A"
        
        print(f"{desc:<30} | {sim_str:<5} | {empty_str:<5} | {tab_score:<7} | {trust_score:.4f} | {'Fraud' if is_fraud else 'Legit'}")
        
    from sklearn.metrics import average_precision_score, roc_auc_score
    if len(set(y_true)) > 1:
        pr_auc = average_precision_score(y_true, y_pred)
        roc_auc = roc_auc_score(y_true, y_pred)
        print('\n--- Real Image Validation Metrics ---')
        print(f'Vision-Fusion PR-AUC:  {pr_auc:.4f}')
        print(f'Vision-Fusion ROC-AUC: {roc_auc:.4f}')

if __name__ == '__main__':
    run()
