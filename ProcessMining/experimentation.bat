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

set repetitions=1 2 3 4 5
set ml_methods=autoencoder variational_autoencoder ocsvm
set fe_window_size=3
set n_clusters=2
set variant=im
set noise_threshold=0.0
set variant_threshold=5000
set normalization_type=zscore
set clustering_type=kmeans
set validation_percentage=0.2

for /D %%p IN ("Results\Training\*") DO (
	del /s /f /q %%p\*.*
	for /f %%f in ('dir /ad /b %%p') do rd /s /q %%p\%%f
	rmdir "%%p" /s /q
)

for /D %%p IN ("Results\Inference\*") DO (
	del /s /f /q %%p\*.*
	for /f %%f in ('dir /ad /b %%p') do rd /s /q %%p\%%f
	rmdir "%%p" /s /q
)

for %%r in (%repetitions%) do (

	mkdir Results\Training\%%r
	mkdir Results\Inference\%%r
	
	for %%m in (%ml_methods%) do (
	
		del /F /Q Input\ELE\TP\*
		copy DataExtraction\Output\%%r\%%m\Inference\TP\* Input\ELE\TP
		mkdir Results\Training\%%r\%%m
		mkdir Results\Inference\%%r\%%m
	
		for %%w in (%fe_window_size%) do (
		
			mkdir Results\Training\%%r\%%m\WS_%%w
			mkdir Results\Inference\%%r\%%m\WS_%%w
		
			for %%c in (%n_clusters%) do (
			
				mkdir Results\Training\%%r\%%m\WS_%%w\NC_%%c
				mkdir Results\Training\%%r\%%m\WS_%%w\NC_%%c\EventLogs\FP
				mkdir Results\Training\%%r\%%m\WS_%%w\NC_%%c\PetriNets
				mkdir Results\Training\%%r\%%m\WS_%%w\NC_%%c\Metrics
				mkdir Results\Training\%%r\%%m\WS_%%w\NC_%%c\Models
				
				mkdir Results\Inference\%%r\%%m\WS_%%w\NC_%%c
				mkdir Results\Inference\%%r\%%m\WS_%%w\NC_%%c\EventLogs\FP
				mkdir Results\Inference\%%r\%%m\WS_%%w\NC_%%c\EventLogs\TP
				mkdir Results\Inference\%%r\%%m\WS_%%w\NC_%%c\ClassificationMetrics
	
				REM Training starts here
				
				del /F /Q Input\ELE\FP\*
				for /D %%p IN ("Output\ELE\*") DO (
					del /s /f /q %%p\*.*
					for /f %%f in ('dir /ad /b %%p') do rd /s /q %%p\%%f
				)
				del /F /Q Input\PMT\EventLogs\*
				del /F /Q Output\PMT\Metrics\*
				del /F /Q Output\PMT\PetriNets\*
				copy DataExtraction\Output\%%r\%%m\Training\FP\* Input\ELE\FP
				
				call process_mining_training %%w %%c %variant_threshold% %validation_percentage%
						
				copy Output\ELE\FP\* Results\Training\%%r\%%m\WS_%%w\NC_%%c\EventLogs\FP
				copy Output\ELE\Models\* Results\Training\%%r\%%m\WS_%%w\NC_%%c\Models
				copy Output\PMT\PetriNets\* Results\Training\%%r\%%m\WS_%%w\NC_%%c\PetriNets
				copy Output\PMT\Metrics\* Results\Training\%%r\%%m\WS_%%w\NC_%%c\Metrics
				
				REM Inference starts here
				
				for /D %%p IN ("Input\PMI\EventLogs\*") DO (
					del /s /f /q %%p\*.*
					for /f %%f in ('dir /ad /b %%p') do rd /s /q %%p\%%f
				)
				del /F /Q Output\ELE\FP\*
				del /F /Q Input\PMI\PetriNets\*
				del /F /Q Input\PMI\Metrics\*
				del /F /Q Input\ELE\FP\*
				copy DataExtraction\Output\%%r\%%m\Inference\FP\* Input\ELE\FP
				
				copy Output\PMT\PetriNets\* Input\PMI\PetriNets
				copy Output\PMT\Metrics\* Input\PMI\Metrics
				
				call process_mining_inference %%w %%c %variant_threshold%
				
				copy Output\ELE\TP\* Results\Inference\%%r\%%m\WS_%%w\NC_%%c\EventLogs\TP
				copy Output\ELE\FP\* Results\Inference\%%r\%%m\WS_%%w\NC_%%c\EventLogs\FP
				copy Output\PMI\ClassificationMetrics\* Results\Inference\%%r\%%m\WS_%%w\NC_%%c\ClassificationMetrics
					
			)
			
		)
		
	)
	
)













