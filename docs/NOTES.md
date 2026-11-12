# Notes that are not the result

## Where the lines come from

Stockport Metropolitan Borough Council purchase-card transactions, data.gov.uk dataset `purchase-card-transactions`, Open Government Licence v3.0. Contains public sector information licensed under the Open Government Licence v3.0.

The purpose map is the reviewed seed from the TrustGate project (https://github.com/VijayChamakuri/trustgate). It collapses raw purpose strings onto the account list. The labelled export matches JudgeGate commit `ff0b8f6b8d3ec119c3f3d2131dc1e8a8b190bb4e`, file `data/stockport_labelled_lines.csv.gz`, SHA-256 `d202870296c57f95014b9f6ad09bc5bc2b3ca68180eaafb37aad3bbc51d3e37b`.

TrustGate is the name of that earlier project. This README uses "the labelled export" instead.

## How a name is scored

The model does not type an account and hope the spelling matches. For each allowed account, the code asks how likely that account name is as the reply, then averages those token probabilities. The average is the rank. The files call this mean token log-probability. A second ranking sums the same probabilities and is not the headline.

`valid.jsonl` is 128 rows copied out of the training file so the trainer can print a validation loss. Those rows stay in the training file. The 2018 lines are in neither file. Do not quote that printed validation loss as the test result.

The losses in `results/mlx_train_curve_2018.json` are the strings the trainer printed. They are not full-precision losses.

## The Mac run

`results/eval_mlx_2018.json` records `ram_gb_memtotal` as null and `gpu` as null. The hardware helper reads Linux `/proc/meminfo` and looks for torch. This Mac has neither, so those fields are empty. `sysctl` on the machine that ran the job reported model `Mac14,2`, brand `Apple M2`, and 8589934592 bytes of memory.

Train time is not a field in the eval JSON. It is the training log's modification time minus its birth time: 2152.5 seconds. The log is `results/mlx_fold_2018_qwen25-15b-instruct-4bit.log`. That file is gitignored.

Peak memory is not in the JSON either. A `footprint` sample every 20 seconds, written to the run log, peaked at `6181.0` MB during scoring (`Footprint: 6181 MB` at 2026-11-12T03:43:27Z). That is one process, not a whole-machine counter. Score time is the JSON field `seconds`: 38911.2. Seconds per line is `seconds_per_example`: 11.488.

The smaller backup model was not used. The eval JSON leaves `fallback_from` empty.

## What `make figures` reads

The top-1 and seen/new charts read the eval JSON files and `results/baselines.json`. The rank chart and the confusion chart read the prediction JSONL files, because the JSON summaries do not store every rank or every pair of booked and predicted accounts. The loss chart reads the training log when it is on disk, and otherwise `results/mlx_train_curve_2018.json`.
