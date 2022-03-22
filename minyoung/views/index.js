
  var wavesurfer = WaveSurfer.create({
    container: '#waveform',
    waveColor: 'black',
    backgroundColor: '#FFGGDD',
    progressColor: 'green',
    cursorColor: 'green',
    cursorWidth:'3',
    hideScrollbar:'ture',
    barHeight: '100',
    height: '400',
    barRadius:'10',
    barGap:'10'
   });


  wavesurfer.on('ready', function(){
    wavesurfer.play();
  });


  wavesurfer.load('../file.wav');
