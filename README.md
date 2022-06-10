## Traffic signs detection model against physical adversarial attacks

Defense method for traffic signs against adversarial attacks.

```
git clone https://github.com/procrastinando/adversarial-attack
cd adversarial-attack
pip install -r requirements.txt
```

### Use:
- Insert dataset in ```traffic_signs/```
- Setup the preparations to create cropped images and a model directory.
- Train the model with the parameters in ```data/```
- Test the results
- Attack the images using a model
- Perform a new testing to corroborate the defense
