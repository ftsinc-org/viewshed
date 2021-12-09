var React = require('react');
var ReactDOM = require('react-dom');
var $ = require('jquery');
var _ = require('lodash');
var querystring = require('querystring');
var BS = require('react-bootstrap');
var Map = require('react-leaflet').Map;
var TileLayer = require('react-leaflet').TileLayer;
var Leaflet = require('react-leaflet');
var Promise = (window && window.Promise) || require('promise-js');
var Forms = require('./forms');

function geoip(){
  return new Promise(function(resolve, reject){
    $.ajax('http://freegeoip.net/json/', {error:reject, success:resolve});
  });
}

function parseFloat(str){
  if(!str){ return ''; }
  var value = window.parseFloat(str);
  if(isNaN(value)){ return ''; }
  return value;
}

var ViewShed = React.createClass({
  getInitialState: function(){
    return {
      lat:51.505,
      lng:-0.09,
      submitDisabled:true,
      radius:10000,
      elevation: 10
    }
  },
  disableSubmit: function(){
    this.setState({submitDisabled: true});
  },
  onValid: _.debounce(function(){
    var values = this.refs.form.getCurrentValues();
    console.log(values);
    this.setState({
      lat: parseFloat(values.lat),
      lng: parseFloat(values.lng),
      submitDisabled: false,
      radius: parseFloat(values.radius),
      elevation: parseFloat(values.elevation)
    });
  }, 750),
  geocodingSelect: function(suggest){
    this.setState({
      lat: suggest.location.lat,
      lng: suggest.location.lng
    });
  },
  mapDoubleClicked: function(event){
    this.setState({
      lat: event.latlng.lat,
      lng: event.latlng.lng
    });
  },
  componentDidMount: function(){
    geoip().then(function(location){
      this.setState({
        lat: location.latitude,
        lng: location.longitude
      });
    }.bind(this));
  },
  onValidSubmit: function(data, resetForm, invalidateForm){
    var url = '/api/v1/viewshed/html?' + $.param({
      lat:data.lat,
      lng:data.lng,
      altitude:data.elevation,
      radius:data.radius
    });
    window.open(url, '_blank').focus();
  },
  render: function(){
    var latlng = [this.state.lat, this.state.lng];
    var submitClass = this.state.submitDisabled ? "btn btn-disabled" : "btn btn-primary";
    var map = this.refs.map;
    if (map){
      map.leafletElement.doubleClickZoom.disable();
      var zoom = map.leafletElement.getZoom() || 12;
      map.leafletElement.setView(latlng, zoom, {animate: true});
    }
    return (
      <BS.Grid fluid={true} style={{height:"100%"}}>
        <Formsy.Form onValidSubmit={this.onValidSubmit} style={{padding:"2px"}} ref="form" onValid={this.onValid} onInvalid={this.disableSubmit}>
          <BS.Row>
            <BS.Col md={6}>
              <Forms.LatInput name="lat" required={true} value={this.state.lat}></Forms.LatInput>
            </BS.Col>
            <BS.Col md={6}>
              <Forms.LngInput name="lng" required={true} value={this.state.lng}></Forms.LngInput>
            </BS.Col>
          </BS.Row>
          <BS.Row>
            <BS.Col md={4}>
              <Forms.TextInput name="elevation" placeholder="Elevation" value={this.state.elevation} required={true}
                validations='isNumeric,isGreaterThanOrEqual:0,isLessThanOrEqual:5000'
                validationErrors={{
                  isNumeric:'elevation must be a number',
                  isGreaterThanOrEqual:'elevation must be greater than 0',
                  isLessThanOrEqual:'elevation must be less than 5000'
                }}>
              </Forms.TextInput>
            </BS.Col>
            <BS.Col md={4}>
              <Forms.TextInput name="radius" placeholder="Radius" value={this.state.radius} required={true}
                validations='isInt,isGreaterThanOrEqual:10,isLessThanOrEqual:100000'
                validationErrors={{
                  isInt:'radius must an whole number',
                  isGreaterThanOrEqual:'radius must be greater than 10',
                  isLessThanOrEqual:'radius must be less than 100000'
                }}>
              </Forms.TextInput>
            </BS.Col>
            <BS.Col md={4}>
              <button style={{width:"100%"}} className={submitClass} type="submit" disabled={this.state.submitDisabled}>Compute Viewshed</button>
            </BS.Col>
          </BS.Row>
        </Formsy.Form>
        <BS.Row style={{height:"100%"}}>
          <BS.Col md={12} style={{height:"100%"}}>
            <Map ref="map" style={{height:"100%"}} onLeafletDblclick={this.mapDoubleClicked}>
              <TileLayer maxZoom={16} url='http://server.arcgisonline.com/ArcGIS/rest/services/NatGeo_World_Map/MapServer/tile/{z}/{y}/{x}' attribution='Tiles &copy; Esri &mdash; National Geographic, Esri, DeLorme, NAVTEQ, UNEP-WCMC, USGS, NASA, ESA, METI, NRCAN, GEBCO, NOAA, iPC'/>
              <Leaflet.Circle radius={this.state.radius} center={latlng}></Leaflet.Circle>
              <Leaflet.Circle radius={10} center={latlng}></Leaflet.Circle>
            </Map>
          </BS.Col>
        </BS.Row>
      </BS.Grid>
    )
  }
});

var ApiViewer = React.createClass({
  componentDidMount: function(){
    var mapCenter = null;
    function onEachFeature(feature, layer){
      //add popup
      if(feature.properties && feature.properties.uiPopupContent) {
        layer.bindPopup(feature.properties.uiPopupContent);
      }
      //set map center
      mapCenter = mapCenter || feature.properties.uiMapCenter;
    }
    window.L.geoJson(window.geoJsonData,{
      onEachFeature:onEachFeature,
      style:{"weight": 4, "opacity": 0.2}
    }).addTo(this.refs.map.leafletElement);
    if(mapCenter){
      this.refs.map.leafletElement.setView([mapCenter[1],mapCenter[0]], 12, {animate: false});
    }
  },
  render: function(){
    return (
      <BS.Grid fluid={true} style={{height:"100%"}}>
        <BS.Row style={{height:"100%"}}>
          <BS.Col md={12} style={{height:"100%"}}>
            <Map ref="map" style={{height:"100%"}}>
              <TileLayer maxZoom={16} url='http://server.arcgisonline.com/ArcGIS/rest/services/NatGeo_World_Map/MapServer/tile/{z}/{y}/{x}' attribution='Tiles &copy; Esri &mdash; National Geographic, Esri, DeLorme, NAVTEQ, UNEP-WCMC, USGS, NASA, ESA, METI, NRCAN, GEBCO, NOAA, iPC'/>
            </Map>
          </BS.Col>
        </BS.Row>
      </BS.Grid>
    );
  }
});

//Render when body loaded
$(function(){
  if(document.getElementById("Viewshed")){
    ReactDOM.render(<ViewShed/>, document.getElementById("Viewshed"));
  }
  else if(document.getElementById("ApiViewer")){
    ReactDOM.render(<ApiViewer/>, document.getElementById("ApiViewer"));
  }
});
//export some things to window
_.extend(window || {}, {
  '_':_,
  '$':$,
  Promise:Promise,
  ReactDOM:ReactDOM
});
