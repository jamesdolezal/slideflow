Search.setIndex({docnames:["activations","appendix","evaluation","extract_tiles","index","pipeline","project","root","slideflow","training","validation"],envversion:{"sphinx.domains.c":2,"sphinx.domains.changeset":1,"sphinx.domains.citation":1,"sphinx.domains.cpp":3,"sphinx.domains.index":1,"sphinx.domains.javascript":2,"sphinx.domains.math":2,"sphinx.domains.python":2,"sphinx.domains.rst":2,"sphinx.domains.std":1,"sphinx.ext.intersphinx":1,"sphinx.ext.todo":2,sphinx:56},filenames:["activations.rst","appendix.rst","evaluation.rst","extract_tiles.rst","index.rst","pipeline.rst","project.rst","root.rst","slideflow.rst","training.rst","validation.rst"],objects:{"slideflow.SlideflowProject":{__init__:[8,1,1,""],add_dataset:[8,1,1,""],associate_slide_names:[8,1,1,""],autoselect_gpu:[8,1,1,""],create_blank_annotations_file:[8,1,1,""],create_blank_train_config:[8,1,1,""],create_hyperparameter_sweep:[8,1,1,""],create_project:[8,1,1,""],evaluate:[8,1,1,""],extract_dual_tiles:[8,1,1,""],extract_tiles:[8,1,1,""],extract_tiles_from_tfrecords:[8,1,1,""],generate_activations_analytics:[8,1,1,""],generate_heatmaps:[8,1,1,""],generate_mosaic:[8,1,1,""],generate_mosaic_from_annotations:[8,1,1,""],generate_tfrecords_from_tiles:[8,1,1,""],generate_thumbnails:[8,1,1,""],get_dataset:[8,1,1,""],load_datasets:[8,1,1,""],load_project:[8,1,1,""],release_gpu:[8,1,1,""],resize_tfrecords:[8,1,1,""],save_project:[8,1,1,""],select_gpu:[8,1,1,""],slide_report:[8,1,1,""],tfrecord_report:[8,1,1,""],train:[8,1,1,""],visualize_tiles:[8,1,1,""]},"slideflow.activations":{ActivationsError:[8,3,1,""],ActivationsVisualizer:[8,0,1,""],Heatmap:[8,0,1,""],TileVisualizer:[8,0,1,""]},"slideflow.activations.ActivationsVisualizer":{__init__:[8,1,1,""],calculate_activation_averages_and_stats:[8,1,1,""],export_activations_to_csv:[8,1,1,""],find_neighbors:[8,1,1,""],generate_activations_from_model:[8,1,1,""],generate_box_plots:[8,1,1,""],get_activations:[8,1,1,""],get_predictions:[8,1,1,""],get_slide_level_categorical_predictions:[8,1,1,""],get_slide_level_linear_predictions:[8,1,1,""],get_tile_node_activations_by_category:[8,1,1,""],get_top_nodes_by_slide:[8,1,1,""],get_top_nodes_by_tile:[8,1,1,""],load_annotations:[8,1,1,""],logistic_regression:[8,1,1,""],map_to_predictions:[8,1,1,""],save_example_tiles_gradient:[8,1,1,""],save_example_tiles_high_low:[8,1,1,""],slide_tile_dict:[8,1,1,""]},"slideflow.activations.Heatmap":{__init__:[8,1,1,""],display:[8,1,1,""],save:[8,1,1,""]},"slideflow.activations.TileVisualizer":{__init__:[8,1,1,""],visualize_tile:[8,1,1,""]},"slideflow.io":{datasets:[8,2,0,"-"],tfrecords:[8,2,0,"-"]},"slideflow.io.datasets":{Dataset:[8,0,1,""],split_tiles:[8,4,1,""]},"slideflow.io.datasets.Dataset":{__init__:[8,1,1,""],get_manifest:[8,1,1,""],get_outcomes_from_annotations:[8,1,1,""],get_rois:[8,1,1,""],get_slide_paths:[8,1,1,""],get_slides:[8,1,1,""],get_tfrecords:[8,1,1,""],get_tfrecords_by_subfolder:[8,1,1,""],get_tfrecords_folders:[8,1,1,""],load_annotations:[8,1,1,""],update_annotations_with_slidenames:[8,1,1,""],update_manifest_at_dir:[8,1,1,""],verify_annotations_slides:[8,1,1,""]},"slideflow.io.tfrecords":{extract_tiles:[8,4,1,""],get_tfrecord_by_index:[8,4,1,""],get_training_and_validation_tfrecords:[8,4,1,""],image_example:[8,4,1,""],join_tfrecord:[8,4,1,""],merge_split_tfrecords:[8,4,1,""],multi_image_example:[8,4,1,""],print_tfrecord:[8,4,1,""],shuffle_tfrecord:[8,4,1,""],shuffle_tfrecords_by_dir:[8,4,1,""],split_patients_list:[8,4,1,""],split_tfrecord:[8,4,1,""],transform_tfrecord:[8,4,1,""],update_tfrecord:[8,4,1,""],update_tfrecord_dir:[8,4,1,""],write_tfrecords_merge:[8,4,1,""],write_tfrecords_multi:[8,4,1,""],write_tfrecords_single:[8,4,1,""]},"slideflow.model":{HyperParameterError:[8,3,1,""],HyperParameters:[8,0,1,""],ManifestError:[8,3,1,""],SlideflowModel:[8,0,1,""],add_regularization:[8,4,1,""]},"slideflow.model.HyperParameters":{__init__:[8,1,1,""],get_model:[8,1,1,""],get_opt:[8,1,1,""],model_type:[8,1,1,""],validate:[8,1,1,""]},"slideflow.model.SlideflowModel":{__init__:[8,1,1,""],evaluate:[8,1,1,""],train:[8,1,1,""]},"slideflow.mosaic":{Mosaic:[8,0,1,""]},"slideflow.mosaic.Mosaic":{__init__:[8,1,1,""],display:[8,1,1,""],focus:[8,1,1,""],save:[8,1,1,""],save_report:[8,1,1,""]},"slideflow.slide":{ExtractionPDF:[8,0,1,""],ExtractionReport:[8,0,1,""],InvalidTileSplitException:[8,3,1,""],JPGslideToVIPS:[8,0,1,""],OpenslideToVIPS:[8,0,1,""],ROIObject:[8,0,1,""],SlideLoader:[8,0,1,""],SlideReader:[8,0,1,""],SlideReport:[8,0,1,""],StainNormalizer:[8,0,1,""],TMAReader:[8,0,1,""],TileCorruptionError:[8,3,1,""],vips2numpy:[8,4,1,""]},"slideflow.slide.ExtractionPDF":{footer:[8,1,1,""]},"slideflow.slide.ExtractionReport":{__init__:[8,1,1,""]},"slideflow.slide.JPGslideToVIPS":{__init__:[8,1,1,""]},"slideflow.slide.OpenslideToVIPS":{__init__:[8,1,1,""],get_best_level_for_downsample:[8,1,1,""],get_downsampled_image:[8,1,1,""],read_region:[8,1,1,""]},"slideflow.slide.ROIObject":{__init__:[8,1,1,""]},"slideflow.slide.SlideLoader":{__init__:[8,1,1,""],loaded_correctly:[8,1,1,""],square_thumb:[8,1,1,""],thumb:[8,1,1,""]},"slideflow.slide.SlideReader":{__init__:[8,1,1,""],annotated_thumb:[8,1,1,""],build_generator:[8,1,1,""],extract_tiles:[8,1,1,""],load_csv_roi:[8,1,1,""],load_json_roi:[8,1,1,""]},"slideflow.slide.SlideReport":{__init__:[8,1,1,""],image_row:[8,1,1,""]},"slideflow.slide.StainNormalizer":{__init__:[8,1,1,""],jpeg_to_jpeg:[8,1,1,""],jpeg_to_rgb:[8,1,1,""],pil_to_pil:[8,1,1,""],rgb_to_rgb:[8,1,1,""],tf_to_rgb:[8,1,1,""]},"slideflow.slide.TMAReader":{__init__:[8,1,1,""],build_generator:[8,1,1,""],extract_tiles:[8,1,1,""]},"slideflow.statistics":{StatisticsError:[8,3,1,""],TFRecordMap:[8,0,1,""],calculate_centroid:[8,4,1,""],gen_umap:[8,4,1,""],generate_combined_roc:[8,4,1,""],generate_histogram:[8,4,1,""],generate_metrics_from_predictions:[8,4,1,""],generate_performance_metrics:[8,4,1,""],generate_roc:[8,4,1,""],generate_scatter:[8,4,1,""],get_centroid_index:[8,4,1,""],normalize_layout:[8,4,1,""],read_predictions:[8,4,1,""],to_onehot:[8,4,1,""]},"slideflow.statistics.TFRecordMap":{__init__:[8,1,1,""],calculate_neighbors:[8,1,1,""],cluster:[8,1,1,""],export_to_csv:[8,1,1,""],filter:[8,1,1,""],from_activations:[8,1,1,""],from_precalculated:[8,1,1,""],get_tiles_in_area:[8,1,1,""],label_by_logits:[8,1,1,""],label_by_slide:[8,1,1,""],label_by_tile_meta:[8,1,1,""],save_2d_plot:[8,1,1,""],save_3d_node_plot:[8,1,1,""],show_neighbors:[8,1,1,""]},"slideflow.util":{ProgressBar:[8,0,1,""],choice_input:[8,4,1,""],dir_input:[8,4,1,""],file_input:[8,4,1,""],float_input:[8,4,1,""],get_slide_paths:[8,4,1,""],global_path:[8,4,1,""],int_input:[8,4,1,""],load_json:[8,4,1,""],make_dir:[8,4,1,""],path_to_ext:[8,4,1,""],path_to_name:[8,4,1,""],read_annotations:[8,4,1,""],read_predictions_from_csv:[8,4,1,""],update_results_log:[8,4,1,""],write_json:[8,4,1,""],yes_no_input:[8,4,1,""]},"slideflow.util.ProgressBar":{__init__:[8,1,1,""]},slideflow:{SlideflowProject:[8,0,1,""],activations:[8,2,0,"-"],model:[8,2,0,"-"],mosaic:[8,2,0,"-"],slide:[8,2,0,"-"],statistics:[8,2,0,"-"],util:[8,2,0,"-"]}},objnames:{"0":["py","class","Python class"],"1":["py","method","Python method"],"2":["py","module","Python module"],"3":["py","exception","Python exception"],"4":["py","function","Python function"]},objtypes:{"0":"py:class","1":"py:method","2":"py:module","3":"py:exception","4":"py:function"},terms:{"01z":6,"3rd":8,"4ha2c":6,"6bc5l":6,"7bf5f":6,"89fcd":6,"case":[1,8,9],"catch":3,"class":[0,1,4,6,8,9],"default":[0,2,3,5,6,8,9],"export":[0,8],"final":[0,5,6,8,9,10],"float":[0,1,3,8,10],"function":[0,2,3,6,8,9],"import":[1,6,8,9,10],"int":[0,3,8,10],"new":[3,4,6,8,10],"return":[0,2,8,9],"true":[0,2,3,6,8,9],"try":8,"while":[1,3,8],For:[0,2,3,6,8,9],Not:8,One:5,SVS:[5,6,8],The:[0,1,2,3,4,5,6,8,9,10],Then:[5,8],There:[0,1,6,9,10],These:[5,6,8,10],Use:6,Used:8,Using:5,Will:[0,2,8,9],With:[0,1],__init__:8,_dir:8,_plot:8,a23a:6,a26x:6,a35b:6,a3co:6,abort:8,about:[6,10],abov:[3,5,8],absolut:8,acceler:8,accept:3,access:2,accomplish:8,accord:[0,3,4,8],account:3,accur:[8,10],accuraci:[3,5,8,9],across:[0,1,2,3,5,6,8,9],action:[3,6,9],activ:[2,5,7],activations_cach:[0,8],activations_export:[0,8],activationserror:8,activationsvisu:[0,8],adam:[6,8,9],add:[3,6,8,9],add_dataset:[6,8],add_regular:8,added:[0,6],addit:[0,3,5,9],addition:0,after:[2,3,5,6,8],against:1,aim:4,algorithm:[1,3,8],all:[0,1,2,3,5,8,9,10],alloc:6,allow:[0,2,8,9],alreadi:[3,8],also:[0,3,5,8],altern:[3,6,8,9],alwai:6,among:8,analys:8,analysi:[4,8],analyt:[4,7,9],analyz:[0,8],ani:[1,3,6,8,10],annot:[0,2,3,5,7,8,9],annotated_thumb:8,annotation_dict:8,annotations_fil:8,anoth:10,anova:[0,8],answer:6,anticip:5,anyth:6,appendix:7,appli:[4,8],applic:9,appropri:[1,8],architectur:[4,5,7,8,9],argument:[0,2,3,6,8,9,10],arrai:[1,3,8,9],artifact:3,asid:[5,10],ask:6,ask_to_merge_subdir:8,assembl:8,assess:[4,5,10],assign:[0,8],assign_slid:8,assist:[5,8],associ:8,associate_slide_nam:8,assum:[0,8],attempt:[0,6,8],auc:[2,5,8],augment:[6,8,9],auto:[6,8],auto_extract:[8,9],autom:5,automat:[0,2,3,5,8,9],autoselect:8,autoselect_gpu:8,avail:[0,1,6,8,9],averag:[0,3,8,10],avg:[6,8,9],avoid:1,axi:[0,8],b83l:6,backend:[4,8],background:[3,5],balanc:[7,8,9],balance_by_categori:[1,6,8,9],balance_by_pati:[1,9],balanced_train:[6,8,9],balanced_valid:[6,8,9],ball_tre:8,bar:8,bar_length:8,base:[1,8],base_level_dim:8,basi:[1,3,8,9],basic:[5,8],batch:[1,5,8,9],batch_fil:[6,8,9],batch_siz:[6,8,9],batch_train:[6,8,9],becaus:1,been:[0,3,4,5,6,8,9],befor:[2,3,5,6,8,9],begin:[3,5,6,7],behavior:[0,8],being:[2,8],belong:[1,8],below:[2,3,5,6,8,9],best:[1,2,5,8,10],better:0,between:[1,8,10],bia:[1,5,10],binari:5,bit:6,black:8,blank:[0,2,6,8],bool:[0,2,3,8,9],bootstrap:[8,9,10],border:8,both:[1,5,8,10],box:[5,8],boxplot:8,bracket:6,braf:3,bright:[3,8],browser:9,brute:8,buffer:[2,3,8],build:[4,8],build_gener:8,built:4,bundl:[6,9],cach:[0,8],calcul:[0,2,3,5,8],calculate_activation_averages_and_stat:8,calculate_centroid:8,calculate_neighbor:8,calcult:8,call:[6,8,9],can:[0,2,3,5,6,8,9,10],capit:6,categor:[0,5,8],categori:[0,1,2,3,6,8,9],cateogri:8,caus:[3,8],center:8,cento:4,centroid:[0,8],certain:[5,8,10],chang:[6,10],checkpoint:[2,8,9],child:8,choic:8,choice_input:8,choos:[0,1,3,5,8,10],chosen:[6,9,10],ckpt:[2,8,9],claim:8,classifi:8,classmethod:8,clear:[1,3],click:5,clinic:6,cluster:8,cmap:[0,8],cnn:4,code:[4,9],cohort:[7,8],collect:[6,8],color:[0,2,8],colormap:[0,2,8],colorspac:[2,3,8],column:[0,2,3,6,8,9],combin:[5,8,9,10],come:[5,10],command:[3,7],commenc:5,comment:6,common:8,compar:[0,8],comparison:8,compat:[8,9],complet:[1,2,5,8,10],complex:[0,8],compon:4,compress:[3,8],comput:[0,8],config:[6,8,9],config_fil:8,configur:[1,3,7,8,9],confirm:[3,8],connect:[1,9],consid:[1,3,8],consider:10,consist:8,constitu:0,construct:8,contain:[3,5,6,8,9],content:[5,7],continu:[5,8,9],conver:3,convert:[1,8],convolut:[1,4,8,9],convolution:8,coordin:[0,4,8],copi:[2,3,8],core:[1,3,8],correctli:8,correspond:[0,2,3,6,8],corrupt:[3,8],cosin:8,cost:[0,8],could:6,count:[3,8],counter_text:8,cours:10,cpkt:8,cpu:3,creat:[0,2,6,7,8,10],create_blank_annotations_fil:8,create_blank_train_config:8,create_hyperparameter_sweep:[6,8,9],create_on_invalid:8,create_project:8,creation:5,crop:8,cross:[5,8,10],csv:[0,6,8,9],ctrl:5,cuda:8,current:[4,8],custom:[0,4],data:[1,2,3,6,7,8,9,10],data_dir:8,data_directori:8,dataset:[0,2,3,5,7,9,10],dataset_nam:6,dataset_with_slidenam:8,decod:8,decreas:9,def:6,default_valu:8,defin:9,delet:[3,8],delete_til:[3,8],denot:6,depend:[5,8],describ:[0,3,4,5,9],descript:8,design:[6,8],desir:[1,5,8],despit:[8,10],dest_tile_px:8,destin:[0,8],detail:[4,5,6,9],detect:[3,6,8],determin:[2,3,8,10],develop:4,dicionari:8,dict:[0,8],dictionari:[0,2,3,8,9],differ:[0,1,3,5,8,9,10],dimens:[0,8],dimension:[0,5,8],dir_input:8,directli:[0,1,2,8,10],directori:[0,2,3,5,6,8,9],discard:[3,5,8],discuss:6,displai:[0,2,8],dispos:1,divid:8,divisor:[3,8],document:[0,3,4,9],doe:[1,5,8],done:[3,5],downsampl:[3,8],downsample_level:8,dpi:8,dtype:8,dual:8,dual_extract:8,due:[8,10],duplic:8,dure:[0,3,5,8,9,10],dx1:6,dynam:8,each:[0,1,2,3,4,5,6,8,9],earli:9,early_stop:[6,8,9],early_stop_method:[8,9],early_stop_pati:[6,8,9],earmark:5,easi:[4,8],easiest:[0,6],easili:6,edit:6,editor:5,effect:1,effici:8,egfr:6,either:[0,1,2,3,5,6,8,9,10],elig:8,ema_observ:8,ema_smooth:8,embed:8,empti:[0,8],enabl:[3,8],enable_downsampl:[3,8],encod:8,end:[4,10],ending_v:8,endtext:8,ensur:[8,10],enter:6,entir:[5,10],entri:[0,2,6,8],environment:[6,8],epoch:[1,5,8,9],equal:[1,2,3,8,9,10],error:8,establish:[1,3,5,6,8],eta:8,etc:[2,5,8,9],eval:6,eval_k_fold:[2,8],evalu:[0,3,5,6,7,8],event:8,everi:[8,9,10],examin:[4,5,6],exampl:[0,1,2,3,4,6,8,9],except:[1,8],excess:1,exclud:[0,2,8],exclude_slid:8,excud:8,execut:[3,4,5,7,8,9],exist:[3,6,8],expand:[0,8],expect:[1,2,8],experiment:8,explicit:8,explicitli:[3,6,8,9],exponenti:8,export_activations_to_csv:8,export_fold:8,export_full_cor:8,export_to_csv:8,expos:1,extend:8,extens:[6,8],extern:[2,10],extract:[0,1,2,5,6,7,8,9,10],extract_dual_til:8,extract_s:8,extract_til:[3,6,8,9],extract_tiles_from_tfrecord:8,extractionpdf:8,extractionreport:8,fail:8,fall:8,fals:[0,2,3,8,9],featur:[0,5,10],feature_label:8,figur:[0,8],file:[0,2,3,5,6,8,9,10],file_input:8,filenam:[0,6,8,9],filetyp:8,fill:8,filter:[0,2,6,7,8,9],filter_blank:[0,2,3,8],find:[0,5,8],find_neighbor:8,fine:[1,5],finetune_epoch:[6,8,9],finish:[2,5,9,10],first:[5,6,8,9,10],five:1,fix:[6,8,9,10],fixedlenfeatur:8,flag:[6,8],flatten:1,flexibl:8,flip:[8,9],float16:8,float32:8,float_input:8,fly:3,focu:8,focus_filt:[0,8],focus_nod:8,fold:[2,5,6,8,9,10],folder:[6,8],follow:[5,6,9,10],footer:8,forc:8,force_gpu:8,force_recalcul:8,force_upd:8,format:[3,5,6,8],found:[8,9],fp16:[6,8],fp32:8,fraction:[3,6,8,9],free:8,freed:[2,8],frequent:8,from:[0,1,2,3,5,8,9,10],from_activ:8,from_precalcul:8,frozen:9,full:[8,9],fulli:[0,1,3,8,9],funtion:8,further:[0,3,4,5],futur:[1,8],gain:0,gen_umap:8,gener:[0,1,3,5,6,7,8,10],generaliz:[1,5,10],generate_activations_analyt:[0,8],generate_activations_from_model:8,generate_box_plot:[0,8],generate_combined_roc:8,generate_heatmap:[2,3,6,8],generate_histogram:8,generate_metrics_from_predict:8,generate_mosa:[0,3,6,8],generate_mosaic_from_annot:8,generate_performance_metr:8,generate_roc:8,generate_scatt:8,generate_tfrecords_from_til:8,generate_thumbnail:8,get:[0,6,8],get_activ:[0,8],get_best_level_for_downsampl:8,get_centroid_index:8,get_dataset:8,get_downsampled_imag:8,get_manifest:8,get_model:8,get_opt:8,get_outcomes_from_annot:8,get_predict:[0,8],get_roi:8,get_slid:8,get_slide_level_categorical_predict:8,get_slide_level_linear_predict:8,get_slide_path:8,get_tfrecord:8,get_tfrecord_by_index:[0,8],get_tfrecords_by_subfold:8,get_tfrecords_fold:8,get_tile_node_activations_by_categori:8,get_tiles_in_area:8,get_top_nodes_by_slid:[0,8],get_top_nodes_by_til:8,get_training_and_validation_tfrecord:8,github:8,given:[0,1,2,3,5,6,8,9],global:[8,10],global_path:8,goal:4,good:[3,10],gpu:[2,6,8],graph:8,grayspac:[7,8],grayspace_fract:[3,8],grayspace_threshold:[3,8],greater:1,greatli:8,green:[2,8],grid:[0,8],groovi:5,group:[3,6,8,10],guid:[5,8],happen:1,has:[0,1,3,4,5,6,8,9,10],have:[0,1,2,3,5,6,8,9,10],hdd:[2,3,8],header:[0,2,3,6,8,9],header_i:8,header_x:8,heatmap:[4,7,8],height:8,help:[4,5,6,8],helper:8,here:4,hidden:[1,9],hidden_lay:[6,8,9],hidden_layer_width:[8,9],hide:8,high:[0,2,4,5,8],highest:[0,8],highlight:[0,8],histogram:[5,8],histolog:[1,4],histori:8,hot:8,how:[0,1,4,6,8,10],howev:[0,1,3],hpsweep0:6,hpv:0,hsv:[3,8],http:[8,9],hue:[3,8],hue_shift:8,human:8,hurt:1,hyperparamet:[1,2,5,7,8,10],hyperparametererror:8,ideal:10,identifi:[6,8],ignor:[8,9],imag:[0,1,2,3,4,5,8,9],image_dict:8,image_exampl:8,image_jpeg:8,image_jpg:8,image_raw:8,image_row:8,image_s:8,image_str:8,imagenet:[1,8,9],impact:[0,8],implement:[1,4,8],improv:[2,3,5,8],imshow:[2,8],incept:1,inceptionv3:[4,5],includ:[0,1,2,3,4,5,8,9],increas:[0,8],index:[7,8],indic:[2,8,10],individu:8,inform:[2,6,8],inher:1,inherit:8,initi:[1,5,6,8],input:[2,3,6,7,8,9],input_arrai:8,input_directori:8,input_fold:8,input_shap:8,input_typ:8,insid:[3,8],insight:[0,10],instanc:[0,8,9,10],instead:[0,8,9,10],institut:10,int_input:8,integr:[3,8],intend:9,interact:[2,6,8],interest:[0,5,8],intermedi:8,intern:[3,8],interpol:[2,8],interpret:8,interv:5,introduc:[0,8],introduct:7,intuit:4,invalid:8,invalidtilesplitexcept:8,involv:5,irrespect:[2,8],issu:[0,8],item:8,iter:[2,8,9,10],its:[1,5,8],jame:6,join_tfrecord:8,jpeg:[3,8],jpeg_str:8,jpeg_to_jpeg:8,jpeg_to_rgb:8,jpg:[6,8],jpgslidetovip:8,json:[2,6,8,9,10],just:[8,9],k_fold_it:[8,9],kd_tree:8,kei:[3,8],kera:[1,4,8,9],kfold3:6,kind:0,know:5,l2_weight:[8,9],label:[0,1,2,3,5,6,8,9],label_by_logit:8,label_by_slid:8,label_by_tile_meta:8,larg:1,last:10,later:[6,8],layer:[1,3,5,7,8,9],layout:8,leadtext:8,learn:[0,1,3,5,8,9,10],learning_r:[6,8,9],least:[2,8],left:5,length:[2,8],lenienc:8,less:1,let:1,letter:6,level:[0,1,2,4,5,8,10],libvip:8,like:[1,2,5,6,8,9],limit:[0,8],line:[3,6],linear:[0,8,9],link:6,list:[0,2,3,8,9],load:[2,5,6,8,9],load_annot:8,load_csv_roi:8,load_dataset:8,load_json:8,load_json_roi:8,load_project:8,loaded_correctli:8,local:[6,8],localhost:9,locat:[6,8],lock:[8,9],log:[8,10],log_frequ:8,logdir:9,logist:8,logistic_regress:8,logit:[0,2,8],logit_cmap:[2,8],longer:8,look:[1,6,8],loop:[8,9],loos:[3,8],loss:[5,6,8,9],low:[0,2,8],low_memori:[0,8],lowest:8,macenko:[3,8],machin:3,made:[0,6,8],magic:10,magnif:5,mai:[0,2,3,4,5,6,8,9,10],main:[6,8],make:[0,6,8],make_dir:8,manag:6,mani:[0,1,2,3,4,5,6,8,9,10],manifest:8,manifesterror:8,manner:8,manual:[3,6,8,9],map:[2,4,7,8,9],map_slid:[0,8],map_to_predict:8,mapped_i:8,mapped_x:8,mask:8,mask_width:8,match:[6,8],matplotlib:[2,8],max:[8,9],max_percentil:8,max_tiles_per_slid:[0,2,8,9],maxim:[2,3,8],maximum:8,mean:1,measur:[5,9],medium:[0,2,8],memori:[0,2,3,8],merg:8,merge_split_tfrecord:8,merge_subdir:8,meta:8,metadata:8,method:[0,1,2,3,8,9],metric:[5,8,9,10],micro:[3,8],microarrai:8,micron:[3,5,8,9],mid:10,might:6,min_dist:8,min_percentil:8,min_tiles_per_slid:[2,8,9],minimum:[2,6,8,9],miss:[3,8],model:[0,2,4,6,7,9,10],model_nam:8,model_path:8,model_typ:[0,8,9],modul:[7,8],monitor:[7,8],more:[0,1,4,5,6,8,9,10],mosaic:[4,7],mosaic_filenam:[0,8],most:[0,8,10],move:8,mpp:8,much:10,multi:8,multi_choic:8,multi_image_exampl:8,multi_input:8,multi_outcom:[8,9],multipl:[5,6,8,9],multithread:8,must:[2,5,6,8,9,10],mutant:6,mutation_statu:3,n_cluster:8,n_compon:8,n_neighbor:8,name:[0,3,6,8,9],nearbi:[0,8],nearest:8,nearestneighbor:8,nearli:3,necessari:[3,8],need:[4,5,6,8,9,10],neg:6,neighbor:8,neighbor_av:8,neighbor_slid:8,nest:8,network:[4,5,10],neural:[4,5,10],next:[1,3],no_bal:[1,6,8,9],nodal:[0,8],node1:8,node2:8,node3:8,node:[0,8],node_exclus:8,node_method:8,node_threshold:8,non:[3,6,8],none:[0,2,3,6,8,9,10],noramlizer_sourc:8,normal:[0,2,7,8,9],normalize_layout:8,normalizer_sourc:[0,2,3,8,9],normalizer_strategi:[8,9],note:[3,7,8,9,10],notic:1,now:1,num_cat:8,num_gpu:8,num_til:8,num_tiles_x:[0,8],num_unique_neighbor:8,number:[0,1,2,5,8,9,10],number_avail:8,numpi:8,object:[0,2,6,8,9],observ:8,occur:[1,5],offer:10,often:5,old:8,old_feature_descript:8,onc:[0,2,3,5,6,9,10],one:[0,5,8,10],onehot:8,onli:[0,1,2,3,5,8,9],onto:[0,8],open:[5,6,9],openslid:8,openslidetovip:8,oppos:[0,5,8],opt:9,optim:[1,5,6,8,9],optimal_til:8,option:[1,2,3,6,8,9,10],optionanl:8,order:[0,1,3,5,8],organ:[4,8],orient:8,origin:[8,10],other:[0,1,3,6,8,9],otherwis:[1,8],our:[1,3],out:[3,5,6],outcom:[0,2,5,6,8,9],outcome_head:[0,2,6,8,9],outcome_label:[0,8],outcome_typ:8,outlier:8,output:[1,8,9],output_directori:8,output_fil:8,output_fold:8,outsid:[3,8],over:[1,3,8,10],overal:[3,5],overarch:4,overfit:10,overhead:3,overlai:[0,2,5,8],overlap:[3,8],overrid:10,overview:[4,6,7],own:[5,8],packag:[4,9],page:7,param:8,paramet:[0,2,3,6,8,9],parent:8,parser:8,particular:0,pass:[0,2,3,6,8,9,10],path:[0,2,3,6,8,9],path_str:8,path_to_ext:8,path_to_nam:8,patient:[1,3,5,6,8,9,10],patients_dict:8,pattern:[0,8],pb_id:8,pdf:[3,8],penultim:[0,1,2,5,8],per:[1,2,3,6,8,9,10],percent:[5,8],percent_matching_categori:8,percentag:[3,8,10],perform:[0,1,2,3,4,5,7,8,10],period:10,permit:8,phase:5,physic:[1,8],pick:10,pil:8,pil_to_pil:8,pipelin:[3,4,6,7,8],pixel:[3,5,8,9],pkl:[0,8],plan:[3,6,7,8,9],pleas:9,plot:[0,5,8,9],png:[0,3],point:[5,8],point_meta:8,polygon:5,pool:[1,6,8,9],poor:[1,5,10],popul:[1,6],posit:[0,6],possess:8,post:1,practic:[1,3],pre:[1,8],precalcul:8,precis:6,predict:[0,1,2,4,5,8,9],predict_on_ax:[0,8],prediction_filt:8,predictions_fil:8,prefer:10,preload:[2,3,8],prelogit:2,prepar:[6,7,8],presenc:8,present:8,preserv:8,press:[5,6],pretrain:[8,9],preval:1,prevent:8,previous:[8,9],print:8,print_tfrecord:8,prior:3,process:[2,8],produc:1,progress:[8,9],progressbar:8,project:[0,2,3,4,5,7,8,9,10],project_fold:8,promin:1,prompt:8,prone:3,proport:[8,10],provid:[0,2,3,4,5,6,8,9],publish:1,pull:0,pyramid:8,python3:6,python:[4,6,8],qualiti:5,question:6,qupath:[5,6],qupath_roi:5,rais:8,random:[8,9,10],randomli:[1,8],rang:1,rapid:8,ras:3,rate:[5,8,9],rather:[6,8],raw:[3,8],read:[0,2,3,5,8],read_annot:8,read_onli:8,read_predict:8,read_predictions_from_csv:8,read_region:8,readabl:8,readi:1,real:[1,3,8],realtim:[8,9],recalcul:8,recommend:[2,3,5,8,10],record:[5,8],red:[2,8],reduc:[1,3,5,8,10],reduct:[0,5,8],refer:3,reflect:[2,8],regard:[6,8],region:[5,8],regress:8,regular:[8,9],reinhard:[3,8],relative_margin:8,relative_s:8,releas:8,release_gpu:8,relev:[8,9],remain:8,remind:6,remov:8,repeat:10,replac:[0,5,8],report:[7,8,10],repres:[1,5,8,9],requir:[3,8,9],reserv:10,reset:8,resiz:8,resize_tfrecord:8,resnet:[4,5],resolut:[0,2,5,8],respect:[3,8],restrict:[0,8],restrict_outcom:8,restrict_predict:[0,8],result:[1,3,6,7,8,10],results_dict:8,results_log:[8,9],results_log_path:8,resum:8,resume_train:[8,9],reveal:8,revers:8,reverse_select_gpu:8,review:[3,4],rgb:[3,8],rgb_to_rgb:8,risk:10,roc:[4,5,8,9],roi:[2,3,6,7,8],roi_dir:8,roi_list:8,roi_method:[3,8],roiobject:8,root:[3,6,8],root_dir:8,rotat:[8,9],roughli:1,row:8,run:[3,5,6],run_project:[6,9],sai:1,same:[1,2,8,10],sampl:[3,8],satur:[3,8],save:[0,2,3,5,6,8,9,10],save_2d_plot:8,save_3d_node_plot:[0,8],save_dir:8,save_example_tiles_gradi:8,save_example_tiles_high_low:8,save_fold:8,save_project:8,save_report:8,save_tfrecord:[3,8],save_til:[3,8],scale:8,scan:8,scatter:[5,8,9],script:[5,6,9],search:[2,6,7,8],second:[5,8,9],section:[5,6,8],see:[3,4,6,8,9],seen:8,segreg:10,select:[0,1,2,3,5,6,7,8,9],select_gpu:8,self:[0,2,3,6,8,9],sensit:8,separ:[2,3,5,8,9,10],sequenti:[2,3,8],seri:6,serv:6,set:[0,1,2,5,7,8,9,10],setup:[5,6],sever:[3,6,10],sfp:[0,2,3,6,9],shape:8,share:6,shift:8,should:[1,2,5,6,8],show:[2,5,8],show_count:8,show_eta:8,show_neighbor:8,show_predict:[0,8],show_roi:[2,8],show_tile_meta:8,shown:3,shuffl:8,shuffle_tfrecord:8,shuffle_tfrecords_by_dir:8,sign:10,signatur:8,signific:[3,8],significantli:[0,8],silent:8,similarli:0,simultan:[8,9],singl:[2,5,8,9],single_thread:[2,8],six:5,size:[0,1,2,3,5,8,9],size_px:[2,8],size_um:[2,8],skip:[2,3,8],skip_extract:[3,8],skip_missing_roi:[3,8],skip_thumb:[2,8],slide1:8,slide:[0,1,2,3,4,5,6,7,9,10],slide_annot:8,slide_categori:8,slide_filt:8,slide_label:8,slide_method:8,slide_node_dict:8,slide_path:8,slide_percentag:8,slide_predict:8,slide_report:[3,8],slide_tile_dict:8,slideflow:[0,1,2,3,4,6,8,9,10],slideflowmodel:8,slideflowproject:[0,2,3,4,6,7,9],slideload:8,slidenam:8,slideread:8,slidereport:8,slides_dir:8,small:8,smooth:[2,8],sne:5,softmax:[1,9],solicit:8,some:[3,4,6],someth:6,sometim:9,sort:8,sourc:[0,2,3,4,6,7,9,10],source_tile_px:8,source_tile_um:8,space:[0,8],spars:1,sparse_categorical_crossentropi:[6,8,9],spatial:[2,8],specif:[6,8],specifi:[0,1,2,3,5,6,8,9],speed:[2,3,5,8],spend:9,split:[5,8],split_fract:8,split_nam:8,split_patients_list:8,split_tfrecord:8,split_til:8,spot:8,squar:[5,8],square_thumb:8,stain:[3,8],stainnorm:8,standard:[1,4],start:[0,6,8,9],starting_epoch:[8,9],starting_v:8,stat:[0,8],statist:[0,7],statisticserror:8,stats_root:8,step:[0,3,6,7,8,10],sthall:8,still:1,stop:9,storag:[3,8],store:[3,5,6,8,9,10],str:8,straightforward:10,strategi:[0,2,3,6,8,9],stream:1,strength:3,strict:[0,8],stride:[2,3,8],stride_div:[3,8],string:[0,8,9],structur:8,subdirectori:8,subfold:8,submitter_id:6,suboptim:3,subsampl:8,subsequ:8,subset:[8,9,10],summar:8,summari:3,superior:3,superv:8,supervis:[0,4,8],suppli:[3,8,9,10],support:4,suppos:1,sure:6,surround:8,svs:[2,6],sweep:[5,8,9],syntax:[6,9],take:8,taken:[0,8,10],target:[3,6,8],tcga:[6,8],tensorboard:[8,9],tensorflow:[4,8,9],tessel:[5,8],test:[1,2,3,4,5,6,8,9,10],text:8,tf_to_rgb:8,tfrecord:[0,2,3,5,6,7,9,10],tfrecord_dict:8,tfrecord_dir:8,tfrecord_fil:8,tfrecord_report:[3,8],tfrecordmap:[0,8],than:[1,8,9],thi:[0,1,2,3,4,6,8,9,10],third:[0,5],thread:[2,8],three:[1,2,5,8,10],threshold:[3,8],through:[1,2,3,6,8,9],throughout:[1,10],thu:[2,8],thumb:8,thumbnail:[2,8],tile1:8,tile2:8,tile3:8,tile:[0,1,2,5,6,7,8,9,10],tile_filt:8,tile_meta:8,tile_px:[3,8,9],tile_select:8,tile_um:[3,8,9],tile_zoom:8,tilecorruptionerror:8,tiles_dir:8,tilevisu:8,time:[1,3,5,6,8,10],tissu:8,titl:8,tma:[3,8],tmaread:8,to_extract:6,to_onehot:8,todo:8,too:[0,8],tool:[4,5,8],toplayer_epoch:[6,8,9],total:8,toward:1,train:[0,1,3,4,6,7,8,10],train_acc:[8,9],train_tfrecord:8,trainable_lay:[8,9],trained_model:[0,2],transfer:9,transform:8,transform_tfrecord:8,translat:8,translation_dict:8,truli:1,tsv:[6,8,9],tumor:[1,3,8],tune:5,tutori:6,two:[0,5,6,8,9],type:[1,6,8],ubuntu:4,umap:[0,5,8],umap_cach:[0,8],umap_export:[0,8],umap_filenam:[0,8],umap_meta:8,unbal:1,unbalanc:1,uncom:[3,6],uniqu:6,unit:8,unless:1,unlik:10,updat:[1,8],update_annotations_with_slidenam:8,update_manifest_at_dir:8,update_results_log:8,update_tfrecord:8,update_tfrecord_dir:8,upon:6,use:[0,1,2,3,4,5,6,8,9,10],use_activations_cach:8,use_centroid:8,use_float:[0,8],use_fp16:8,use_optimal_til:8,used:[1,2,3,5,6,8,9,10],useful:6,user:[1,6,8],uses:[2,6,8],using:[0,2,4,5,6,8,9,10],util:7,vahadan:[3,8],val:8,val_acc:[8,9],val_batch_s:8,val_loss:[8,9],valid:[3,5,6,7,8,9],valid_choic:8,valid_rang:8,validate_on_batch:[8,9],validation_annot:[8,9],validation_dataset:[8,9],validation_filt:[8,9],validation_fract:[8,9,10],validation_k_fold:[8,9,10],validation_log:8,validation_step:[8,9],validation_strategi:[7,8,9],validation_target:[7,8,9],validation_tfrecord:8,valu:[0,2,3,8,9],vari:0,variabl:[6,8,9],varieti:[5,8],variou:[0,5],vastli:[2,3,8],verif:8,verifi:8,verify_annotations_slid:8,vgg16:[1,4,5],via:[0,8,9,10],view:[0,3],vip:8,vips2numpi:8,visual:[0,7,8],visualize_til:8,vmtouch:[2,3,8],wai:[0,4,6,8,10],wait:[5,9],walk:1,want:3,web:9,weight:[1,8,9],well:[4,5,8],were:8,what:[1,6],when:[0,1,2,3,6,8,10],where:[2,6,8,10],wherea:1,whether:[1,2,3,6,8,9,10],which:[0,1,2,3,4,5,6,8,9,10],whitespac:[5,7,8],whitespace_fract:[3,8],whitespace_on_ax:[0,8],whitespace_threshold:[3,8],whole:[5,8],width:[1,2,3,8,9],window:8,within:[0,1,3,6,8],without:[6,8,9],work:7,workflow:5,world:1,would:[2,6,8,9],wrap:[2,8],wrapper:8,write:8,write_json:8,write_tfrecords_merg:8,write_tfrecords_multi:8,write_tfrecords_singl:8,written:8,x_lower:8,x_upper:8,xception:[1,4,6,8,9],y_lower:8,y_pred:8,y_true:8,y_upper:8,yes:[6,8],yes_no_input:8,yet:[3,8],you:[0,1,2,3,4,5,6,9,10],your:[1,2,3,4,5,6,8,9,10],zero:[2,8,9],zoom:8},titles:["Layer activations","Appendix","Evaluation","Tile extraction","Introduction","Pipeline Overview","Setting up a Project","Slideflow Documentation","Source","Training","Validation Planning"],titleterms:{activ:[0,8],analyt:5,annot:6,appendix:1,architectur:1,balanc:1,begin:9,cohort:10,command:6,configur:[6,10],creat:5,data:5,dataset:[6,8],document:7,evalu:[2,10],execut:6,extract:3,filter:3,gener:2,grayspac:3,heatmap:[2,5],hyperparamet:9,indic:7,input:1,introduct:4,layer:0,map:[0,5],model:[1,5,8],monitor:9,mosaic:[0,5,8],normal:3,note:1,overview:5,perform:9,pipelin:5,plan:10,prepar:[5,9],project:6,report:3,result:5,roi:5,select:10,set:6,slide:8,slideflow:7,slideflowproject:8,sourc:8,statist:8,step:5,tabl:7,tfrecord:8,tile:3,train:[5,9],util:8,valid:10,validation_strategi:10,validation_target:10,visual:5,whitespac:3,work:0}})