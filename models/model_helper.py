from .MBConv import MBConvBlock
from .MHSA import MHSABlock


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic',
        'mean': (0.485, 0.456, 0.406), 'std': (0.229, 0.224, 0.225),
        'classifier': 'head',
        **kwargs
    }


default_cfgs = {
    'MetaFG_0': _cfg(),
    'MetaFG_1': _cfg(),
    'MetaFG_2': _cfg(),
}


def make_blocks(stage_index, depths, embed_dims, img_size, dpr, extra_token_num=1, num_heads=8, mlp_ratio=4.,
                stage_type='conv'):
    stage_name = f'stage_{stage_index}'
    blocks = []
    for block_idx in range(depths[stage_index]):
        stride = 2 if block_idx == 0 and stage_index != 1 else 1
        in_chans = embed_dims[stage_index] if block_idx != 0 else embed_dims[stage_index - 1]
        out_chans = embed_dims[stage_index]
        image_size = img_size if block_idx == 0 or stage_index == 1 else img_size // 2
        drop_path_rate = dpr[sum(depths[1:stage_index]) + block_idx]
        if stage_type == 'conv':
            blocks.append(MBConvBlock(ksize=3, input_filters=in_chans, output_filters=out_chans,
                                      image_size=image_size, expand_ratio=int(mlp_ratio), stride=stride,
                                      drop_connect_rate=drop_path_rate))
        elif stage_type == 'mhsa':
            blocks.append(MHSABlock(input_dim=in_chans, output_dim=out_chans,
                                    image_size=image_size, stride=stride, num_heads=num_heads,
                                    extra_token_num=extra_token_num,
                                    mlp_ratio=mlp_ratio, drop_path=drop_path_rate))
        else:
            raise NotImplementedError("We only support conv and mhsa")
    return blocks
