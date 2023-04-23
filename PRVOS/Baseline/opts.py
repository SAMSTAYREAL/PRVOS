import argparse

def get_args_parser():
    parser = argparse.ArgumentParser(description="DeepLab-ResNet Network")
    parser.add_argument("--arch", type=str, default='', help='model')

    parser.add_argument("--desc", type=str, default='')
    parser.add_argument("--eval", action='store_true')
    parser.add_argument("--eval_first", action='store_true')
    parser.add_argument("--mode", type=str, default='train_davis')
        
    parser.add_argument("--init_lr", type=float, default=1e-6, help="init lr")  # 1e-4
    parser.add_argument("--batch_size", type=int, default=6, help="batch size")
    parser.add_argument("--test_batch_size", type=int, default=1, help="batch size for eval")
        
    parser.add_argument("--img_size", type=int, default=320, help="image size")
    parser.add_argument("--max_epoch", type=int, default=50, help="lr decay epoch") # 20
    parser.add_argument("--decay_epochs", type=int, default=[30,40], nargs='+', help='50 80 100')  # 10,15
        
    parser.add_argument("--optimizer", type=str, default='adam')
    parser.add_argument("--lr_decay", type=float, default=0.5, help="lr decay")
    parser.add_argument("--save_every", type=int, default=0, help="save_every")
        
    parser.add_argument("--dataset", type=str, default='refer-yv-2019')  
    parser.add_argument("--test_dataset", type=str, default='refer-yv-2019')  
    parser.add_argument("--splits", type=str, default='train', help='test splits')
    
    parser.add_argument("--max_N", type=int, default=0, help="lr decay epoch")
    parser.add_argument("--max_skip", type=int, default=2, help="max_skip for video")

    parser.add_argument("--checkpoint", type=str, default='')  
    parser.add_argument("--epoch", type=int, default=-1, help="resume training from the specified epoch number.")
    
    parser.add_argument("--finetune", type=bool, default=False)
    
      
    return parser.parse_args()