class ConfigDs(object):
    def __init__(self, params,model_path='models_ds/default.json'):

        # with open(model_path[:-4] + "json") as f:
        #     params = json.load(f)

        self.model_name = model_path
        
        
        self.expType = params['name']
        self.sample_rate = params['resamp']
        self.clip_samples = int(self.sample_rate * params['duration'])
        self.mel_bins = params['mels'] #64
        self.fmin = params['f_min'] # 50
        self.fmax = params['f_max'] #14000
        self.window_size = params['fft'] # 1024
        self.hop_size = params['hop'] #320
        self.window = 'hann'
        self.pad_mode = 'reflect'
        self.center = True # center the array before applying padding
        self.device = 'cuda'
        self.ref = 1.0 # refrence value for power-to-db
        self.amin = 1e-10
        self.top_db = params['db'] #None
        # self.freeze_base = params['freeze_base']
        self.duration = params['duration']
        self.labels = params['class_list'] #['background', 'shot']
        self.lb_to_idx = {lb: idx for idx, lb in enumerate(self.labels)}
        self.idx_to_lb = {idx: lb for idx, lb in enumerate(self.labels)}
        self.classes_num = len(self.labels)

        self.loss_type = params['loss_type']
        self.augmentations =  params['augmentations']
        self.learning_rate = params['learning_rate']
        self.batch_size = params['batch_size']
        self.epochs = params['epochs']
        

    def update(self, args): 
        self.mel_bins = args.mel_bins #64
        self.fmin = args.fmin
        self.fmax = args.fmax #14000
        self.window_size =args.window_size
        self.hop_size =  args.hop_size
        self.window = args.window
        self.amin = args.amin 

        self.loss_type = args.loss_type
        self.augmentations =  args.augmentations 
        self.learning_rate = args.learning_rate
        self.batch_size =  args.batch_size