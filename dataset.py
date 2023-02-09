import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image, ImageEnhance

transform_type_dict = dict(
    Brightness=ImageEnhance.Brightness,
    Contrast=ImageEnhance.Contrast,
    Sharpness=ImageEnhance.Sharpness,
    Color=ImageEnhance.Color
)


def extract_frames(video_path, interval=1):
    """
    Args:
        video_path (str): denote the video path.
        interval (int):  sampling interval.
    Return:
        frames (list): the elements in list are numpy.ndarray.
        status (list): [video_path, FLAG]
    """
    cap = cv2.VideoCapture(video_path)
    imgs = []
    if not cap.isOpened():
        status = [video_path, "Opening Failed"]
        return imgs, status
    index = 0
    while True:
        if interval > 1:
            cap.set(cv2.CAP_PROP_POS_FRAMES, index)
            index += interval
        ret, frame = cap.read()
        if not ret:
            break
        imgs.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    if len(imgs) == 0:
        return imgs
    return imgs


class VideoDataset:
    def __init__(
            self,
            sub_meta=None,
            cl=None,
            data_file=None,
            transform=transforms.ToTensor(),
            target_transform=lambda x: x,
            random_select=False,
            num_segments=None
    ):
        self.cl = cl
        if data_file is None:
            self.meta = sub_meta
        else:
            self.meta = [x.strip().split(' ') for x in open(data_file)]
        self.transform = transform
        self.target_transform = target_transform
        self.random_select = random_select
        self.num_segments = num_segments
        self.current_video_path = None
        self.current_video_frames = []

    def _load_image(self, vid_path, idx):
        """
        Get an image from the list of frames loaded for a video, load a new video
        when the video name has changed
        Args:
            vid_path: the path of the video
            idx: the index of the frame

        Returns:
            the frame
        """
        if self.current_video_path != vid_path:
            self.current_video_path = vid_path
            self.current_video_frames = extract_frames(vid_path)
        return Image.fromarray(self.current_video_frames[idx - 1])

    def __getitem__(self, i):
        """
        Get a sequence of snippets from a video
        Args:
            i: index in data list

        Returns:
            sequence of snippets from a video
        """
        full_path = self.meta[i][0]
        num_frames = int(self.meta[i][1])
        num_segments = self.num_segments
        if self.random_select and num_frames > 8:  # random sample
            average_duration = num_frames // num_segments
            frame_id = np.multiply(list(range(num_segments)), average_duration)
            frame_id = frame_id + np.random.randint(average_duration, size=num_segments)
        else:
            tick = num_frames / float(num_segments)
            frame_id = np.array([int(tick / 2.0 + tick * x) for x in range(num_segments)])
        frame_id = frame_id + 1  # idx >= 1
        img_group = []
        for k in range(self.num_segments):
            img = self._load_image(full_path, frame_id[k])
            img = self.transform(img)
            img_group.append(img)
        img_group = torch.stack(img_group, 0)
        target = self.target_transform(int(self.meta[i][2]) if self.cl is None else self.cl)
        return img_group, target

    def __len__(self):
        return len(self.meta)


class SetDataManager:
    """
    Handler for getting a dataset
    """

    def __init__(self, image_size, n_way, n_support, n_query, num_segments, n_episode=100):
        super(SetDataManager, self).__init__()
        self.image_size = image_size
        self.n_way = n_way
        self.batch_size = n_support + n_query
        self.n_episode = n_episode
        self.trans_loader = TransformLoader(image_size)
        self.num_segments = num_segments

    def get_data_loader(self, data_file, aug):
        """
        Get a dataset, mainly handles the difference between train/val/test
        Args:
            data_file: list of videos in the subset
            aug: use augmentation?

        Returns:
            the data loader
        """
        transform = self.trans_loader.get_composed_transform(aug)
        dataset = SetDataset(
            data_file,
            self.batch_size,
            transform,
            random_select=aug,
            num_segments=self.num_segments
        )
        sampler = EpisodicBatchSampler(len(dataset), self.n_way, self.n_episode)
        data_loader_params = dict(batch_sampler=sampler, num_workers=8, pin_memory=True)
        data_loader = torch.utils.data.DataLoader(dataset, **data_loader_params)
        return data_loader


class TransformLoader:
    """
    Build preprocessing steps
    """

    def __init__(self, image_size, normalize_param=None, jitter_param=None):
        if normalize_param is None:
            normalize_param = dict(mean=[0.376, 0.401, 0.431], std=[0.224, 0.229, 0.235])
        if jitter_param is None:
            jitter_param = dict(Brightness=0.4, Contrast=0.4, Color=0.4)
        self.image_size = image_size
        self.normalize_param = normalize_param
        self.jitter_param = jitter_param

    def parse_transform(self, transform_type):
        """
        Interpret transform params
        Args:
            transform_type: the type of transformation

        Returns:
            the transform method
        """
        if transform_type == 'ImageJitter':
            return ImageJitter(self.jitter_param)
        method = getattr(transforms, transform_type)
        if transform_type in ['RandomResizedCrop', 'CenterCrop']:
            return method(self.image_size)
        if transform_type == 'Resize':
            return method([int(self.image_size * 1.15), int(self.image_size * 1.15)])
        if transform_type == 'Normalize':
            return method(**self.normalize_param)
        return method()

    def get_composed_transform(self, aug=False):
        """
        Combine the transforms
        Args:
            aug: use augmentation?

        Returns:
            a single transform performing all steps
        """
        if aug:
            transform_list = ['RandomResizedCrop', 'ImageJitter', 'RandomHorizontalFlip', 'ToTensor', 'Normalize']
        else:
            transform_list = ['Resize', 'CenterCrop', 'ToTensor', 'Normalize']
        transform_funcs = [self.parse_transform(x) for x in transform_list]
        transform = transforms.Compose(transform_funcs)
        return transform


class SetDataset:
    """
    Dataset for an episode
    """

    def __init__(self, data_file, batch_size, transform, random_select=False, num_segments=None):
        self.video_list = [x.strip().split(' ') for x in open(data_file)]
        self.cl_list = np.zeros(len(self.video_list), dtype=int)
        for i in range(len(self.video_list)):
            self.cl_list[i] = self.video_list[i][2]
        self.cl_list = np.unique(self.cl_list).tolist()

        self.sub_meta = {}
        for cl in self.cl_list:
            self.sub_meta[cl] = []

        for x in range(len(self.video_list)):
            root_path = self.video_list[x][0]
            num_frames = int(self.video_list[x][1])
            label = int(self.video_list[x][2])
            self.sub_meta[label].append([root_path, num_frames])

        self.sub_dataloader = []
        sub_data_loader_params = dict(
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,  # use main thread only or may receive multiple batches
            pin_memory=False

        )
        for cl in self.cl_list:
            sub_dataset = VideoDataset(
                sub_meta=self.sub_meta[cl],
                cl=cl,
                transform=transform,
                random_select=random_select,
                num_segments=num_segments
            )
            self.sub_dataloader.append(torch.utils.data.DataLoader(sub_dataset, **sub_data_loader_params))

    def __getitem__(self, i):
        return next(iter(self.sub_dataloader[i]))

    def __len__(self):
        return len(self.cl_list)


class EpisodicBatchSampler(object):
    """
    Defines what can be sampled in an episode
    """

    def __init__(self, n_classes, n_way, n_episodes):
        self.n_classes = n_classes
        self.n_way = n_way
        self.n_episodes = n_episodes

    def __len__(self):
        return self.n_episodes

    def __iter__(self):
        for _ in range(self.n_episodes):
            yield torch.randperm(self.n_classes)[:self.n_way]


class ImageJitter(object):
    """
    Randomly jitter the image, improve generalization
    """

    def __init__(self, transform_dict):
        self.transforms = [(transform_type_dict[k], transform_dict[k]) for k in transform_dict]

    def __call__(self, img):
        out = img
        rand_tensor = torch.rand(len(self.transforms))
        for i, (transformer, alpha) in enumerate(self.transforms):
            r = alpha * (rand_tensor[i] * 2.0 - 1.0) + 1
            out = transformer(out).enhance(r).convert('RGB')
        return out


def to_tensor(data):
    if isinstance(data, torch.Tensor):
        return data
    elif isinstance(data, np.ndarray):
        return torch.from_numpy(data)
    elif isinstance(data, int):
        return torch.LongTensor([data])
    elif isinstance(data, float):
        return torch.FloatTensor([data])
    else:
        raise TypeError('type {} cannot be converted to tensor.'.format(
            type(data)))


class SimpleDataManager:
    def __init__(self, image_size, batch_size, num_segments=None):
        super(SimpleDataManager, self).__init__()
        self.batch_size = batch_size
        self.num_segments = num_segments
        self.trans_loader = TransformLoader(image_size)

    def get_data_loader(self, data_file, aug):  # parameters that would change on train/val set
        transform = self.trans_loader.get_composed_transform(aug)
        dataset = VideoDataset(
            data_file=data_file,
            transform=transform,
            random_select=aug,
            num_segments=self.num_segments
        )
        data_loader_params = dict(batch_size=self.batch_size, shuffle=True, num_workers=8, pin_memory=True)
        data_loader = torch.utils.data.DataLoader(dataset, **data_loader_params)

        return data_loader
