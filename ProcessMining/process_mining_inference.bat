:: %1: window size, %2: number of clusters, %3: variant threshold

:: Useful commands:

:: cd <directory>
:: copy path\to\file destination\path
:: xcopy path\to\dirs destination\path /E
:: rmdir "<dir_name>" /s /q
:: ren path\to\file <name>
:: del /F /Q path\to\file 


set fe_window_size=%1
set n_clusters=%2
set normalization_type=zscore
set clustering_type=kmeans
set variant_threshold=%3
set normalization_type=zscore
set clustering_type=kmeans

python event_log_extraction.py inference %fe_window_size% %normalization_type% %clustering_type% %n_clusters%

copy Output\ELE\TP\* Input\PMI\EventLogs\TP
copy Output\ELE\FP\* Input\PMI\EventLogs\FP
				
python process_mining_inference.py
				













