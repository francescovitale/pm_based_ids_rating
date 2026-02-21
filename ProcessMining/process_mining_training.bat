:: %1: window size, %2: number of clusters, %3: variant threshold, %4: validation percentage

:: Useful commands:

:: cd <directory>
:: copy path\to\file destination\path
:: xcopy path\to\dirs destination\path /E
:: rmdir "<dir_name>" /s /q
:: ren path\to\file <name>
:: del /F /Q path\to\file 

:: Options:

:: fe_window_size=<integer>
:: clustering_type=[kmeans, gmm, agglomerative]
:: n_clusters=<integer>
:: variant=[im, ilp]
:: noise_threshold=<float>

set fe_window_size=%1
set n_clusters=%2
set variant_threshold=%3
set variant=im
set noise_threshold=0.0
set normalization_type=zscore
set clustering_type=kmeans
set validation_percentage=%4
			
python event_log_extraction.py training %fe_window_size% %normalization_type% %clustering_type% %n_clusters%

copy Output\ELE\FP\* Input\PMT\EventLogs
				
python process_mining_training.py %variant% %noise_threshold% %validation_percentage%
				













