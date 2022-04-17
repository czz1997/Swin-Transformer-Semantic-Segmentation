# model settings
norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    type='EncoderDecoder',
    pretrained=None,
    backbone=dict(
        type='SwinTransformer',  # backbone type
        embed_dim=96,  # the following will be passed to swin transformer's constructor as arguments
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        mlp_ratio=4.,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.3,
        ape=False,
        patch_norm=True,
        out_indices=(0, 1, 2, 3),
        use_checkpoint=False),
    decode_head=dict(
        type='UPerHead',  # type of decode head, see mmseg/models/decode_heads
        in_channels=[96, 192, 384, 768],  # input channels of each decode heads
        in_index=[0, 1, 2, 3],  # indices of feature maps
        pool_scales=(1, 2, 3, 6),
        channels=512,
        dropout_ratio=0.1,  # dropout ratio before the final classification layer
        num_classes=19,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(  # loss function for decode heads
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    auxiliary_head=dict(
        type='FCNHead',  # type of auxiliary head, see mmseg/models/decode_heads
        in_channels=384,
        in_index=2,
        channels=256,
        num_convs=1,  # number of convs in FCNHead, typically 1
        concat_input=False,  # whether to concat input and conv output before classification layer
        dropout_ratio=0.1,
        num_classes=19,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4)),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))
