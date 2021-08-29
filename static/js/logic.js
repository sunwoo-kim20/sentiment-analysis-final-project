function clickedFace(choice) {
    // call route to give vote to flask 
    if (choice === 'Positive') {
        d3.json('/positive_update').then(unused => {})
    }
    if (choice === 'Negative') {
        d3.json('/negative_update').then(unused => {})
    }
    // update tweet
    updateTweet();
}

function updateTweet() {
    d3.json("/apicall").then(data =>{
        d3.select(".tweetholder").text(data.tweet)
        console.log(data)
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

url = 'https://publish.twitter.com/oembed?https://twitter.com/Interior/status/'

function embedTweet() {
    d3.json("/apicall").then(data =>{
        var tweetID = data.tweetID.toString()
        d3.select(".tweetholder").append("a")
            .attr("href", "url"+tweetID)

        twttr.widgets.load(
            document.getElementByClassName("tweetholder")
        )
        
    })
}

// grab a tweet when webpage is opened
updateTweet();