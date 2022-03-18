# MLPerf system description checker

MLPerf system description checker

## Usage

To check a system description json file for compliance:

```sh
python3 -m mlperf_logging.system_desc_checker FILENAME USAGE RULESET
```

Currently, USAGE in ["training"] and RULESET in ["0.6.0", "0.7.0"] are supported.

Prints SUCCESS when no issues were found. Otherwise, will print FAILURE with error details.

## Tested software versions
Tested and confirmed working using the following software versions:

Python 2.7.18
Python 3.7.7
