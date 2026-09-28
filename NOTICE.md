# Attribution

fastsem-jev builds on the frozen Qwen direct option-logit decision design used by
[SemIf](https://github.com/TheoLeeCJ/SemIf), by Theodore Lee (TheoLeeCJ).
The validation, direct prompt, model-loading helpers, and option readout are derived
from the local research implementation based on SemIf commit
`ca3ba65f142967030ecb453346e94d6f476a69df`. The upstream MIT copyright notice
is preserved in LICENSE. Token compression and its experiments are fastsem-jev work.

Qwen3.5-4B weights are downloaded separately from
[Qwen](https://huggingface.co/Qwen/Qwen3.5-4B) under their own Apache-2.0 license.
No model weights or third-party benchmark question text are bundled.
JevBench is a separate Benchmark Heaven project; these local results are not an
official JevBench submission, score, or endorsement. Native-method results describe
our Qwen3.5-4B migrations, not the authors' original model/service deployments.
