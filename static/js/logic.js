function clickedFace(choice) {
    var data = d3.select(".tweetholder").data()[0]
    console.log(data)
    // call route to give vote to flask 
    if (choice === 'Positive') {
        $.post("/positive_update", data);
        location.reload();
    }
    if (choice === 'Negative') {
        $.post( "/negative_update", data);
        location.reload();
    }
    if (choice === 'Neutral') {
        $.post("/neutral_update", data);
        location.reload();

    }

    // update tweet
    // updateTweet();
    // location.reload();
}

// function noSentiment(choice) {
//     var data = d3.select(".tweetholder").data()[0]
//     choice == 'Neutral'
//     updateTweet();
// }

function updateTweet() {
    d3.json("/load_tweet").then(data =>{
        // give data to the tweet and change the tweet text

        d3.select(".tweetholder").data([data]).text(data.tweet)
        console.log(data)
        console.log("update tweet")
        // make prediction 
        // var prediction;
        // if (data.predicted_sentiments_rd >= .5) {
        //     prediction = "POSITIVE"
        // }
        // else {
        //     prediction = "NEGATIVE"
        // }
        // updatePrediction(prediction)
    })
}

// function updatePrediction(choice) {
//     d3.select(".modelPredict").text(`Our model predicts this tweet has a ${choice} sentiment.`)
// }

// grab a tweet when webpage is opened
updateTweet();