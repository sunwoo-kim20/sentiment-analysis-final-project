function clickedFace(choice) {
    updateTweet();
    updatePrediction(choice)
}

function updateTweet() {
    d3.json("/apicall").then(data =>{
        d3.select(".tweetholder").text(data.tweet)
        var prediction;
        if (data.sentiment >= .5) {
            prediction = "POSITIVE"
        }
        else {
            prediction = "NEGATIVE"
        }

        updatePrediction(prediction)
    })
}

function updatePrediction(choice) {
    d3.select(".modelPredict").text(`Our model predicts this tweet has a ${choice} sentiment.`)
}

updatePrediction("");
updateTweet();