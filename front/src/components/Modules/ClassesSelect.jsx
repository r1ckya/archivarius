import React from 'react';
import {useLocation} from "react-router-dom";
import {Button, Col, Form, FormControl, Nav, Navbar, NavDropdown, Row, Table} from "react-bootstrap";

function useQuery() {
  return new URLSearchParams(useLocation().search);
}
let ClassesSelect = function (props) {
  console.log(props)
  return (
    <Form.Group  as={Row} controlId="exampleForm.ControlSelect1">
      <Form.Label column sm="2">Класс документа</Form.Label>
      <Col sm="10">
        <Form.Control onChange={props.handleChange}  as="select" name={'doc_cls'} value={props.type}>
          <option value={"undefined"}>Любой</option>
            {props.classes.map(function(object, i){
              return <option value={object}>{object}</option>;
            })}
        </Form.Control>
      </Col>
    </Form.Group>
  )

}

export default ClassesSelect;