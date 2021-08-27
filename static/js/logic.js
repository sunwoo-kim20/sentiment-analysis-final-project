var counter = 0;

function clickedFace(choice) {
    updateTweet();
    updatePrediction(choice)
}

function updateTweet() {
    counter += 1;
    d3.select(".tweet").text(counter);
}

function updatePrediction(choice) {
    d3.select("#modelPredict").text(`Our model predicts this tweet has a ${choice} sentiment.`)
}

updatePrediction("");
updateTweet("");