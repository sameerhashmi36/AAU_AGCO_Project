from ultralytics import YOLO

def train_yolo11():
    model = YOLO('yolo11m.pt')  # Ensure model is available or it'll auto-download
    # model.train(data='./merged_dataset.yaml', epochs=20, batch=8, workers = 12)
    model.train(
        data        = './merged_dataset.yaml',
        epochs      = 5,
        batch       = 4,          
        device      = 0,
        cache       = 'ram',      # cache in memory
        workers     = 8,          # more data loaders
        amp         = True,       # mixed precision
        rect        = True,       # rectangular batches
        mosaic      = 0.0,        # turn off heavy aug
        mixup       = 0.0,
        auto_augment= 'none',
        hsv_h       = 0.0,
        hsv_s       = 0.0,
        hsv_v       = 0.0,
        perspective = 0.0,
        project     = 'runs/train',
        name        = 'merged11m_fast',
        exist_ok    = True,
    )

if __name__ == '__main__':
    train_yolo11()
