@echo off

echo ============================================================
echo Cloning 17 passing candidates for manual review
echo ============================================================
echo.

if not exist "data\repos" mkdir "data\repos"

echo [1/17] ANTsX/ANTsTorch (CV - Medical)
git clone --depth 50 https://github.com/ANTsX/ANTsTorch.git data\repos\ANTsTorch

echo [2/17] ternaus/midv-500-models (CV - Document Segmentation)
git clone --depth 50 https://github.com/ternaus/midv-500-models.git data\repos\midv-500-models

echo [3/17] MR-HosseinzadehTaher/BenchmarkTransferLearning (CV - Medical)
git clone --depth 50 https://github.com/MR-HosseinzadehTaher/BenchmarkTransferLearning.git data\repos\BenchmarkTransferLearning

echo [4/17] nick8592/lane-detection-unet-ncnn (CV - Autonomous)
git clone --depth 50 https://github.com/nick8592/lane-detection-unet-ncnn.git data\repos\lane-detection-unet-ncnn

echo [5/17] AidinHamedi/Pytorch-Img-Classification-Trainer-V2 (CV)
git clone --depth 50 https://github.com/AidinHamedi/Pytorch-Img-Classification-Trainer-V2.git data\repos\Pytorch-Img-Classification-Trainer-V2

echo [6/17] aminK8/TaxaDiffusion (CV - Generative)
git clone --depth 50 https://github.com/aminK8/TaxaDiffusion.git data\repos\TaxaDiffusion

echo [7/17] AnInsomniacy/tracknet-series-pytorch (CV - Sports)
git clone --depth 50 https://github.com/AnInsomniacy/tracknet-series-pytorch.git data\repos\tracknet-series-pytorch

echo [8/17] MatN23/AdaptiveTrainingSystem (NLP - LLM)
git clone --depth 50 https://github.com/MatN23/AdaptiveTrainingSystem.git data\repos\AdaptiveTrainingSystem

echo [9/17] gokhaneraslan/chatterbox-finetuning (NLP - TTS)
git clone --depth 50 https://github.com/gokhaneraslan/chatterbox-finetuning.git data\repos\chatterbox-finetuning

echo [10/17] KuchikiRenji/vall-e (NLP - TTS)
git clone --depth 50 https://github.com/KuchikiRenji/vall-e.git data\repos\vall-e

echo [11/17] iflytek/cino (NLP)
git clone --depth 50 https://github.com/iflytek/cino.git data\repos\cino

echo [12/17] Ahmed-El-Zainy/coding_llama2 (NLP - LLM)
git clone --depth 50 https://github.com/Ahmed-El-Zainy/coding_llama2.git data\repos\coding_llama2

echo [13/17] soran-ghaderi/torchebm (Generative)
git clone --depth 50 https://github.com/soran-ghaderi/torchebm.git data\repos\torchebm

echo [14/17] LucaGeminiani00/Diffusion-Distillation-WL (Generative - TimeSeries)
git clone --depth 50 https://github.com/LucaGeminiani00/Diffusion-Distillation-WL.git data\repos\Diffusion-Distillation-WL

echo [15/17] jorshi/flucoma-torch (Audio)
git clone --depth 50 https://github.com/jorshi/flucoma-torch.git data\repos\flucoma-torch

echo [16/17] haowei01/pytorch-examples (Tabular/Ranking)
git clone --depth 50 https://github.com/haowei01/pytorch-examples.git data\repos\pytorch-examples

echo [17/17] ewijaya/gflownet-peptide (Drug Discovery)
git clone --depth 50 https://github.com/ewijaya/gflownet-peptide.git data\repos\gflownet-peptide

echo.
echo ============================================================
echo Done. All 17 repos cloned to data\repos\
echo.
echo Next: Review each repo, then run the AST classifier:
echo   python scripts/ast_classifier.py --repos data\repos\ --output data\manifests\
echo ============================================================
pause
