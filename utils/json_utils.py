import re

def clean_json_text(text: str) -> str:
    """
    Cleans raw LLM response strings by stripping markdown code block wrappers
    (e.g. ```json ... ``` or ``` ... ```) and trimming whitespace so json.loads
    can reliably parse it without throwing JSONDecodeError.
    """
    if not text:
        return ""
    cleaned = text.strip()
    
    # Strip markdown fences if present
    pattern = r"^```(?:json)?\s*([\s\S]*?)\s*```$"
    match = re.match(pattern, cleaned, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()
        
    return cleaned

def format_pydantic_validation_error(exc) -> str:
    """
    Extracts field-level details from Pydantic ValidationError for backend logging.
    """
    if hasattr(exc, "errors") and callable(exc.errors):
        formatted_errors = []
        for err in exc.errors():
            field_path = ".".join(str(p) for p in err.get("loc", []))
            msg = err.get("msg", "Validation error")
            inp = repr(err.get("input", None))
            err_type = err.get("type", "unknown")
            formatted_errors.append(f"Field: '{field_path}' | Error: {msg} | Received: {inp} | Type: {err_type}")
        return "\n  - ".join([""] + formatted_errors)
    return str(exc)

def normalize_grounded_analysis(raw_data: dict) -> dict:
    """
    Normalizes raw Gemini Vision output dictionaries to match the canonical GroundedAnalysisResult schema.
    Handles field name casing differences (camelCase to snake_case), alternate keys (e.g. category vs name),
    string scene values, null lists/dicts, float/out-of-bounds coordinates, and Google 2D box arrays.
    """
    if not isinstance(raw_data, dict):
        return {}

    # Unwrap top-level wrappers if present
    if 'result' in raw_data and isinstance(raw_data['result'], dict):
        raw_data = raw_data['result']
    elif 'data' in raw_data and isinstance(raw_data['data'], dict):
        raw_data = raw_data['data']

    normalized = {}

    # 1. Executive / Overall Summary
    overall_sum = (
        raw_data.get('overall_summary') or 
        raw_data.get('overallSummary') or 
        raw_data.get('summary') or 
        raw_data.get('executive_summary') or 
        'Visual analysis completed.'
    )
    normalized['overall_summary'] = str(overall_sum)

    # 2. Scene Understanding
    raw_scene = raw_data.get('scene')
    if isinstance(raw_scene, str):
        normalized['scene'] = {
            'category': 'General Scene',
            'environment': raw_scene,
            'primary_activity': 'Observed Scene',
            'summary': raw_scene
        }
    elif isinstance(raw_scene, dict):
        cat = (raw_scene.get('category') or raw_scene.get('scene_category') or raw_scene.get('sceneCategory') or 'General Scene')
        env = (raw_scene.get('environment') or raw_scene.get('sceneEnvironment') or raw_scene.get('environment_description') or 'Environment observed')
        act = (raw_scene.get('primary_activity') or raw_scene.get('primaryActivity') or raw_scene.get('activity') or 'Primary activity')
        summ = (raw_scene.get('summary') or raw_scene.get('sceneSummary') or raw_scene.get('description') or 'Scene summary')
        normalized['scene'] = {
            'category': str(cat),
            'environment': str(env),
            'primary_activity': str(act),
            'summary': str(summ)
        }
    else:
        normalized['scene'] = {
            'category': 'General Scene',
            'environment': 'Environment observed',
            'primary_activity': 'Primary activity',
            'summary': 'Scene summary'
        }

    # 3. Object Categories & Instances
    raw_objects = raw_data.get('objects') or raw_data.get('object_categories') or raw_data.get('detected_objects') or []
    if not isinstance(raw_objects, list):
        raw_objects = []

    norm_objects = []
    for cat_item in raw_objects:
        if not isinstance(cat_item, dict):
            continue

        name = (cat_item.get('name') or cat_item.get('category') or cat_item.get('category_name') or cat_item.get('label') or 'object')
        
        # Counts
        conf_count = cat_item.get('confirmed_count')
        if conf_count is None:
            conf_count = cat_item.get('confirmedCount')
        if conf_count is None:
            conf_count = cat_item.get('count')
        if conf_count is None:
            conf_count = len(cat_item.get('instances') or []) if isinstance(cat_item.get('instances'), list) else 1

        try:
            conf_count = max(0, int(conf_count))
        except (ValueError, TypeError):
            conf_count = 1

        unc_count = cat_item.get('uncertain_count')
        if unc_count is None:
            unc_count = cat_item.get('uncertainCount')
        try:
            unc_count = max(0, int(unc_count)) if unc_count is not None else 0
        except (ValueError, TypeError):
            unc_count = 0

        # Instances
        raw_instances = cat_item.get('instances')
        if not isinstance(raw_instances, list):
            raw_instances = []

        norm_instances = []
        for idx_i, inst_item in enumerate(raw_instances, start=1):
            if not isinstance(inst_item, dict):
                continue

            inst_id = (inst_item.get('id') or inst_item.get('instance_id') or inst_item.get('instanceId') or f'{name}_{idx_i}')

            # Attributes
            raw_attr = inst_item.get('attributes')
            if not isinstance(raw_attr, dict):
                raw_attr = {}

            norm_attr = {
                'clothing': str(raw_attr.get('clothing') or 'not clearly visible'),
                'clothing_color': str(raw_attr.get('clothing_color') or raw_attr.get('clothingColor') or 'not clearly visible'),
                'pose': str(raw_attr.get('pose') or 'unknown'),
                'action': str(raw_attr.get('action') or 'unknown'),
                'accessories': str(raw_attr.get('accessories') or 'none visible'),
                'object_color': str(raw_attr.get('object_color') or raw_attr.get('objectColor') or 'not clearly visible'),
                'type_or_subtype': str(raw_attr.get('type_or_subtype') or raw_attr.get('typeOrSubtype') or raw_attr.get('type') or 'unknown'),
                'visible_details': str(raw_attr.get('visible_details') or raw_attr.get('visibleDetails') or 'none noted'),
            }

            # Bounding Box
            raw_box = inst_item.get('bounding_box') or inst_item.get('boundingBox') or inst_item.get('box_2d') or inst_item.get('box2d')
            norm_box = None

            if isinstance(raw_box, (list, tuple)) and len(raw_box) == 4:
                # [ymin, xmin, ymax, xmax] -> x_min, y_min, x_max, y_max
                try:
                    vals = [float(v) for v in raw_box]
                    if all(0.0 <= v <= 1.0 for v in vals):
                        vals = [v * 1000.0 for v in vals]
                    y1, x1, y2, x2 = vals
                    norm_box = {
                        'x_min': max(0, min(1000, int(round(min(x1, x2))))),
                        'y_min': max(0, min(1000, int(round(min(y1, y2))))),
                        'x_max': max(0, min(1000, int(round(max(x1, x2))))),
                        'y_max': max(0, min(1000, int(round(max(y1, y2)))))
                    }
                except (ValueError, TypeError):
                    norm_box = None
            elif isinstance(raw_box, dict):
                x_min = raw_box.get('x_min') if raw_box.get('x_min') is not None else raw_box.get('xmin', raw_box.get('xMin'))
                y_min = raw_box.get('y_min') if raw_box.get('y_min') is not None else raw_box.get('ymin', raw_box.get('yMin'))
                x_max = raw_box.get('x_max') if raw_box.get('x_max') is not None else raw_box.get('xmax', raw_box.get('xMax'))
                y_max = raw_box.get('y_max') if raw_box.get('y_max') is not None else raw_box.get('ymax', raw_box.get('yMax'))

                if all(v is not None for v in [x_min, y_min, x_max, y_max]):
                    try:
                        fx1, fy1, fx2, fy2 = float(x_min), float(y_min), float(x_max), float(y_max)
                        if all(0.0 <= v <= 1.0 for v in [fx1, fy1, fx2, fy2]):
                            fx1, fy1, fx2, fy2 = fx1 * 1000.0, fy1 * 1000.0, fx2 * 1000.0, fy2 * 1000.0
                        norm_box = {
                            'x_min': max(0, min(1000, int(round(min(fx1, fx2))))),
                            'y_min': max(0, min(1000, int(round(min(fy1, fy2))))),
                            'x_max': max(0, min(1000, int(round(max(fx1, fx2))))),
                            'y_max': max(0, min(1000, int(round(max(fy1, fy2)))))
                        }
                    except (ValueError, TypeError):
                        norm_box = None

            unc_reason = inst_item.get('uncertainty_reason') or inst_item.get('uncertaintyReason')

            norm_instances.append({
                'id': str(inst_id),
                'attributes': norm_attr,
                'bounding_box': norm_box,
                'uncertainty_reason': str(unc_reason) if unc_reason else None
            })

        norm_objects.append({
            'name': str(name).lower(),
            'confirmed_count': conf_count,
            'uncertain_count': unc_count,
            'instances': norm_instances
        })

    normalized['objects'] = norm_objects
    return normalized
