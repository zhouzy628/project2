_base_ = ["yolov3_mobilenetv2_mstrain-416_300e_coco.py"]

data=dict(
  #sample_per_gpu=24,#12 yyh
  sample_per_gpu=8,
  train=dict(
    dataset=dict(
      ann_file='/content/gdrive/MyDrive/mmdetection/dataset/fruitdata/annotations/train.json',
      img_prefix='/content/gdrive/MyDrive/mmdetection/dataset/fruitdata/train',
      classes=("lemon",)
    )
  ),
  val=dict(
    ann_file='/content/gdrive/MyDrive/mmdetection/dataset/fruitdata/annotations/train.json',
    img_prefix='/content/gdrive/MyDrive/mmdetection/dataset/fruitdata/train',
    classes=("lemon",)
  ),
  test=dict(
    ann_file='/content/gdrive/MyDrive/mmdetection/dataset/fruitdata/annotations/train.json',
    img_prefix='/content/gdrive/MyDrive/mmdetection/dataset/fruitdata/train',
    classes=("lemon",)
  )
)

model=dict(
  bbox_head=dict(
    num_classes=1 #number of classes
  )
)

#decrease loss(from 580)
#使用预训练的检测模型作为梯度下降起点，微调训练
#yolov 300e:原来训练了300轮
load_from="/content/gdrive/MyDrive/mmdetection/dataset/yolov3_mobilenetv2_mstrain-416_300e_coco_20210718_010823-f68a07b3.pth"
# epoch
runner = dict(type='EpochBasedRunner', max_epochs=8) ##30-->8 YYH
##max_epochs 30x10 =300e


# learning rate:模型已经训练的较为接近，需要调学习率达到更好的训练效果
#optimizer = dict(type='SGD', lr=0.003, momentum=0.9, weight_decay=0.0005) #lr:0.003-->0.001 YYH
optimizer = dict(type='SGD', lr=0.001)
#lr初始学习率：0.003


# logging
lr_config = None

# optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# lr_config = dict( #lr_config 控制学习率
#     policy='step',
#     warmup='linear', #warmup:开始时学习率逐渐上升
#     warmup_iters=4000,
#     warmup_ratio=0.0001,
#     step=[24, 28])#从24-28轮降低学习率 从头训练需要

# evaluation = dict(interval=1, metric=['bbox'])#每一轮评价一次指定的性能
# find_unused_parameters = True
# work_dir = './work_dirs/fruit'
# auto_resume = False
# gpu_ids = range(0, 1)



#log_config = dict(interval=50, hooks=[dict(type='TextLoggerHook')]
log_config = dict(interval=25, hooks=[dict(type='TextLoggerHook')])