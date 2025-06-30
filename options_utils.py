def default_options() -> dict:
    return {
        'weights': 'yolo7.pt',
        'cfg': '',
        'hyp': 'data/hyp.scratch.p5.yaml',
        'epochs': 300,
        'batch-size': 16,
        'img-size': [640, 640],
        'resume': False,
        'bucket': '',
        'device': '',
        'local_rank': -1,
        'workers': 8,
        'project': 'runs/train',
        'entity': None,
        'name': 'exp',
        'label-smoothing': 0.0,
        'bbox_interval': -1,
        'save_period': -1,
        'artifact_alias': 'latest',
        'freeze': [0],
        'rect': False,
        'nosave': False,
        'notest': False,
        'noautoanchor': False,
        'evolve': False,
        'cache-images': False,
        'image-weights': False,
        'multi-scale': False,
        'single-cls': False,
        'adam': False,
        'sync-bn': False,
        'exist-ok': False,
        'quad': False,
        'linear-lr': False,
        'upload_dataset': False,
        'v5-metric': False
    }


def merge_default(input: dict) -> dict:
    output = {}
    output.update(default_options())
    output.update(input)
    return output
