param(
 [string]$Mode='AliveStatus',
 [string]$StateDir='',
 [string]$Output='',
 [string]$Message='',
 [string]$Intent='',
 [string]$Text='',
 [string]$Lessons='',
 [string]$LessonId='',
 [string]$ReviewedBy='',
 [string]$Permission='',
 [string]$Challenger='',
 [string]$Incumbent='',
 [string]$Suite='',
 [string]$EncoderManifest='',
 [int]$Steps=400,
 [int]$Seed=52021,
 [double]$RetentionEpsilon=0.0,
 [int]$Port=39061,
 [string]$PythonExecutable=''
)
$ErrorActionPreference='Stop'
$Root=$PSScriptRoot
$Venv=Join-Path $Root '.kev-venv'
$Py=Join-Path $Venv 'Scripts\python.exe'
if ($env:OS -ne 'Windows_NT') { $Py=Join-Path $Venv 'bin/python' }
if (-not [string]::IsNullOrWhiteSpace($PythonExecutable)) { $Py=$PythonExecutable }

if ($Mode -eq 'Install') {
 $Bootstrap=(Get-Command python -ErrorAction Stop).Source
 & $Bootstrap -m venv $Venv
 & $Py -m pip install -r (Join-Path $Root 'requirements.txt')
 & $Py -m pip install -e $Root
 exit $LASTEXITCODE
}
if (-not (Test-Path -LiteralPath $Py)) { throw 'Run .\KEV.ps1 -Mode Install first.' }

$Base=@('-m','kev.uc51a3.cli')
if (-not [string]::IsNullOrWhiteSpace($StateDir)) { $Base+=@('--state-dir',$StateDir) }
if ($Mode -eq 'AliveStatus') { & $Py @Base 'status'; exit $LASTEXITCODE }
if ($Mode -eq 'AliveVerifyLedger') { & $Py @Base 'verify-ledger'; exit $LASTEXITCODE }
if ($Mode -eq 'AliveChat') { & $Py @Base 'chat' '--message' $Message; exit $LASTEXITCODE }
if ($Mode -eq 'AliveTeach') { & $Py @Base 'teach' '--intent' $Intent '--text' $Text; exit $LASTEXITCODE }
if ($Mode -eq 'AliveReview') { & $Py @Base 'review' '--lesson-id' $LessonId '--reviewed-by' $ReviewedBy '--permission' $Permission; exit $LASTEXITCODE }
if ($Mode -eq 'AliveTrain') {
 $Args=@('train-candidate','--output',$Output,'--steps',[string]$Steps,'--seed',[string]$Seed)
 if (-not [string]::IsNullOrWhiteSpace($EncoderManifest)) { $Args+=@('--encoder-manifest',$EncoderManifest) }
 & $Py @Base @Args
 exit $LASTEXITCODE
}
if ($Mode -eq 'Eval') {
 $Args=@('-m','kev.cli','eval','--challenger',$Challenger,'--incumbent',$Incumbent,'--suite',$Suite,'--retention-epsilon',[string]$RetentionEpsilon)
 if (-not [string]::IsNullOrWhiteSpace($Output)) { $Args+=@('--output',$Output) }
 & $Py @Args
 exit $LASTEXITCODE
}
if ($Mode -eq 'Evolve') {
 $Args=@('-m','kev.cli','evolve','--lessons',$Lessons,'--steps',[string]$Steps,'--seed',[string]$Seed,'--retention-epsilon',[string]$RetentionEpsilon)
 if (-not [string]::IsNullOrWhiteSpace($StateDir)) { $Args+=@('--state-dir',$StateDir) }
 if (-not [string]::IsNullOrWhiteSpace($EncoderManifest)) { $Args+=@('--encoder-manifest',$EncoderManifest) }
 & $Py @Args
 exit $LASTEXITCODE
}
if ($Mode -eq 'Doctor') { & $Py -m kev.cli doctor; exit $LASTEXITCODE }
if ($Mode -eq 'AliveDesktop') {
 $Args=@('-m','kev.uc51a3.server','--port',[string]$Port)
 if (-not [string]::IsNullOrWhiteSpace($StateDir)) { $Args+=@('--state-dir',$StateDir) }
 & $Py @Args
 exit $LASTEXITCODE
}
throw "Unknown public-source mode: $Mode"
