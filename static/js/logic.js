function clickedFace(choice) {
    // call route to give vote to flask 
    if (choice === 'Positive') {
        d3.json('/positive_chosen').then(unused => {})
    }
    if (choice === 'Negative') {
        d3.json('/negative_chosen').then(unused => {})
    }
    // update tweet
    updateTweet();
}

function updateTweet() {
    d3.json("/apicall").then(data =>{
        d3.select(".tweetholder").text(data.tweet)
        var prediction;
        console.log(data.sentiment)
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

// grab a tweet when webpage is opened
updateTweet();