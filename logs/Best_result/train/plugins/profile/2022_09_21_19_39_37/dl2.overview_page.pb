?	??{?}@??{?}@!??{?}@	XWWL??@XWWL??@!XWWL??@"z
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails9??{?}@?ߢ??V
@1eQ|@A????U?\?I??Z?{?@Y?(?N?(@r0*֣p=:l?@)      ?=2\
%Iterator::Root::FlatMap[0]::Generator?	?8-@!?x?+m?X@)?	?8-@1?x?+m?X@:Preprocessing2E
Iterator::Root??a?-@!      Y@).?&??1@xQ?齻?:Preprocessing2N
Iterator::Root::FlatMap??V`?-@!?k???X@)???E_A??1??ʋf???:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 2.6% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9XWWL??@I?kt9????Q?s????W@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?ߢ??V
@?ߢ??V
@!?ߢ??V
@      ??!       "	eQ|@eQ|@!eQ|@*      ??!       2	????U?\?????U?\?!????U?\?:	??Z?{?@??Z?{?@!??Z?{?@B      ??!       J	?(?N?(@?(?N?(@!?(?N?(@R      ??!       Z	?(?N?(@?(?N?(@!?(?N?(@b      ??!       JGPUYXWWL??@b q?kt9????y?s????W@?"c
7gradient_tape/model/conv1_2/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter???S+͐?!???S+͐?0"c
7gradient_tape/model/conv1_5/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter}R??ʐ?!o???ˠ?0"c
7gradient_tape/model/conv1_3/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter[?Q???!?u?j'??0"c
7gradient_tape/model/conv1_7/Conv2D/Conv2DBackpropFilterConv2DBackpropFilterზR???!??`?I???0"c
7gradient_tape/model/conv1_4/Conv2D/Conv2DBackpropFilterConv2DBackpropFilteru?5n???!cZ?x?ݴ?0"c
7gradient_tape/model/conv1_6/Conv2D/Conv2DBackpropFilterConv2DBackpropFilterF??odL??!4??????0"e
9gradient_tape/model/deconv1_7/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter5@???[??!;?(;\??0"-
IteratorGetNext/_1_Send?e????!?͚??m??"-
IteratorGetNext/_3_Send?P}O?}?!??o{?G??"c
7gradient_tape/model/conv2_7/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter?;d?c3{?!?	?\?}??0Q      Y@Yx?? S@a\???fUX@q]>??@??y-?'?E?u?"?	
device?Your program is NOT input-bound because only 2.6% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
nono*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQ2"Nvidia GPU (Ampere)(: B 