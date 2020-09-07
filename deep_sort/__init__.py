from .deep_sort import DeepSort


__all__ = ['DeepSort', 'build_tracker']


def build_tracker(cfg, cam2world, obj_height_meters, use_cuda):
    return DeepSort(model_path=cfg.DEEPSORT.REID_CKPT,
                    vehicle_model_path=cfg.DEEPSORT.VEHICLE_REID_CKPT,
                    cam2world=cam2world,
                    obj_height_meters=obj_height_meters,
                    max_depth=cfg.DEEPSORT.MAX_DEPTH,
                    max_dist=cfg.DEEPSORT.MAX_DIST,
                    min_confidence=cfg.DEEPSORT.MIN_CONFIDENCE,
                    nms_max_overlap=cfg.DEEPSORT.NMS_MAX_OVERLAP,
                    max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
                    max_age=cfg.DEEPSORT.MAX_AGE,
                    n_init=cfg.DEEPSORT.N_INIT,
                    nn_budget=cfg.DEEPSORT.NN_BUDGET,
                    use_cuda=use_cuda)