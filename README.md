# Performance Analysis: SteamVR vs. libsurvive

## File Structure
- data
  - exp_type
    - steamvr/libsurvive
        - date+exp num
          - "{number}".txt:
            - number = 1 if drift 
            - number =  1-X if repeatability or static analysis
            - number = 1-6 if dynamic
        
Note: Each point in repeatability experiment is a different exp_num 