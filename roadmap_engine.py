import copy
import numpy as np

FEATURE_NAMES = [
    'monsoonIntensity', 'TopographyDrainage', 'RiverManagement', 'Deforestation',
    'Urbanization', 'ClimateChange', 'DamsQuality', 'Siltation',
    'AgriculturalPractices', 'Encroachments', 'ineffectiveDisasterPreparedness',
    'DrainageSystems', 'Coastal vulnerability', 'Landslides', 'watershades',
    'deteriorating infrastructure', 'population score', 'WetlandLoss',
    'InadequatePlanning', 'PoliticalFactors'
]

IMPROVEMENTS = [
    {
        'id': 'early_warning',
        'name': 'Early Warning & Response Systems',
        'description': 'Build community alert systems, evacuation training and emergency communication to reduce risk from sudden floods.',
        'category': 'Preparedness',
        'cost_usd': 320000,
        'timeline_years': 1,
        'impact': {'ineffectiveDisasterPreparedness': -4}
    },
    {
        'id': 'drainage_upgrade',
        'name': 'Upgrade Urban Drainage Networks',
        'description': 'Improve sewer, canal and stormwater capacity to reduce surface water accumulation during heavy rain.',
        'category': 'Infrastructure',
        'cost_usd': 1250000,
        'timeline_years': 2,
        'impact': {'DrainageSystems': -3}
    },
    {
        'id': 'river_dredging',
        'name': 'River Dredging & Silt Management',
        'description': 'Remove sediment from waterways and restore flow capacity to reduce river overflow and backup.',
        'category': 'Maintenance',
        'cost_usd': 950000,
        'timeline_years': 2,
        'impact': {'Siltation': -3, 'RiverManagement': -1}
    },
    {
        'id': 'wetland_restoration',
        'name': 'Wetland & Floodplain Restoration',
        'description': 'Restore wetlands and natural floodplains to retain excess water and protect communities downstream.',
        'category': 'Nature-based',
        'cost_usd': 820000,
        'timeline_years': 3,
        'impact': {'WetlandLoss': -4, 'Coastal vulnerability': -1}
    },
    {
        'id': 'dam_strengthening',
        'name': 'Dam & Embankment Strengthening',
        'description': 'Reinforce dams, levees and embankments to reduce the chance of structural failures during extreme events.',
        'category': 'Infrastructure',
        'cost_usd': 2200000,
        'timeline_years': 3,
        'impact': {'DamsQuality': -4}
    },
    {
        'id': 'anti_encroachment',
        'name': 'Anti-Encroachment Enforcement',
        'description': 'Clear illegal construction from flood corridors and enforce zoning rules to reduce obstruction of natural flow paths.',
        'category': 'Governance',
        'cost_usd': 410000,
        'timeline_years': 1,
        'impact': {'Encroachments': -3}
    },
    {
        'id': 'urban_planning',
        'name': 'Urban Resilience Planning',
        'description': 'Integrate flood-safe zoning, green infrastructure and stormwater planning in city development.',
        'category': 'Policy',
        'cost_usd': 780000,
        'timeline_years': 2,
        'impact': {'Urbanization': -2, 'InadequatePlanning': -3}
    },
    {
        'id': 'reforestation',
        'name': 'Reforestation & Soil Conservation',
        'description': 'Plant trees and build soil retention systems upstream to reduce erosion and downstream sediment build-up.',
        'category': 'Nature-based',
        'cost_usd': 650000,
        'timeline_years': 3,
        'impact': {'Deforestation': -4, 'Siltation': -2}
    },
    {
        'id': 'coastal_defense',
        'name': 'Coastal Defense & Mangrove Restoration',
        'description': 'Strengthen shorelines with natural and engineered defenses to reduce coastal flood vulnerability.',
        'category': 'Infrastructure',
        'cost_usd': 1400000,
        'timeline_years': 3,
        'impact': {'Coastal vulnerability': -4, 'ineffectiveDisasterPreparedness': -1}
    },
    {
        'id': 'landslide_stabilization',
        'name': 'Landslide Stabilization & Slope Protection',
        'description': 'Stabilize slopes and improve land drainage in vulnerable hillsides to limit landslide-triggered flooding.',
        'category': 'Nature-based',
        'cost_usd': 900000,
        'timeline_years': 2,
        'impact': {'Landslides': -3}
    },
    {
        'id': 'community_training',
        'name': 'Community Training & Preparedness',
        'description': 'Educate local communities on flood preparation, evacuation planning and safe response actions.',
        'category': 'Preparedness',
        'cost_usd': 260000,
        'timeline_years': 1,
        'impact': {'ineffectiveDisasterPreparedness': -2}
    },
    {
        'id': 'governance_reform',
        'name': 'Water Governance & Planning Reform',
        'description': 'Improve policy coordination and planning to reduce political and regulatory barriers to flood resilience.',
        'category': 'Governance',
        'cost_usd': 560000,
        'timeline_years': 2,
        'impact': {'PoliticalFactors': -2, 'InadequatePlanning': -2}
    }
]


def clamp_value(value):
    return max(0, min(14, round(value, 1)))


def apply_improvements(baseline_features, selected_ids):
    updated = copy.deepcopy(baseline_features)
    for imp_id in selected_ids:
        improvement = next((item for item in IMPROVEMENTS if item['id'] == imp_id), None)
        if not improvement:
            continue
        for feature, delta in improvement['impact'].items():
            if feature in updated:
                updated[feature] = clamp_value(updated[feature] + delta)
    return updated


def calculate_risk(model, features_dict, feature_names=FEATURE_NAMES):
    feature_values = [features_dict.get(name, 0) for name in feature_names]
    X = np.array(feature_values).reshape(1, -1)
    prediction = model.predict(X)[0]
    probabilities = model.predict_proba(X)[0]
    class_index = list(model.classes_).index(prediction)
    confidence = round(probabilities[class_index] * 100, 1)
    return {
        'risk_class': str(prediction).strip().lower(),
        'risk_score': round(float(np.max(probabilities)), 3),
        'confidence': confidence,
        'probabilities': {str(label): round(float(score), 3) for label, score in zip(model.classes_, probabilities)}
    }


def evaluate_improvement(model, baseline_features, improvement, feature_names=FEATURE_NAMES):
    improved_features = apply_improvements(baseline_features, [improvement['id']])
    baseline_risk = calculate_risk(model, baseline_features, feature_names)
    improved_risk = calculate_risk(model, improved_features, feature_names)
    risk_reduction = round(baseline_risk['risk_score'] - improved_risk['risk_score'], 4)
    roi = round(risk_reduction / (improvement['cost_usd'] / 1_000_000), 4) if improvement['cost_usd'] > 0 else 0
    return {
        'id': improvement['id'],
        'name': improvement['name'],
        'description': improvement['description'],
        'category': improvement['category'],
        'cost_usd': improvement['cost_usd'],
        'cost_display': f"${improvement['cost_usd']:,}",
        'timeline_years': improvement['timeline_years'],
        'impact': improvement['impact'],
        'risk_reduction': risk_reduction,
        'roi': roi,
        'baseline_risk': baseline_risk,
        'improved_risk': improved_risk
    }


def suggest_roadmap(model, baseline_features, feature_names=FEATURE_NAMES, budget=None):
    candidates = [evaluate_improvement(model, baseline_features, imp, feature_names) for imp in IMPROVEMENTS]
    candidates = [item for item in candidates if item['risk_reduction'] > 0]
    candidates.sort(key=lambda x: (x['roi'], x['risk_reduction']), reverse=True)
    budget_usd = float(budget) if budget is not None and budget > 0 else None
    if budget_usd is None:
        return {
            'recommendations': candidates,
            'budget_usd': 0,
            'budget_remaining_usd': 0,
            'total_cost_usd': 0
        }

    selected = []
    remaining = budget_usd
    total_cost = 0
    for item in candidates:
        if item['cost_usd'] <= remaining:
            selected.append(item)
            remaining -= item['cost_usd']
            total_cost += item['cost_usd']
    return {
        'recommendations': selected,
        'budget_usd': budget_usd,
        'budget_remaining_usd': round(remaining, 2),
        'total_cost_usd': total_cost
    }
